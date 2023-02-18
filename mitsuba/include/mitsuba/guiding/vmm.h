#pragma once
#include "common.h"
#include <vector>
#include <atomic>
#include <mutex>

MTS_NAMESPACE_BEGIN

class VMM {
public:

    // TODO expand the value range of kappa
    constexpr static Float MIN_KAPPA = 1e-2;
    constexpr static Float MAX_KAPPA = 1e+4;
    constexpr static Float REDUCTION_POWER = -0.65;
    constexpr static int BATCH_SIZE = 64;
    constexpr static int NUM_INITIAL_COMPONENTS = 16;

    inline explicit VMM(int numComponents = NUM_INITIAL_COMPONENTS) {
        components.reserve(numComponents);
        for (int k = 0; k < numComponents; ++k) {
            // TODO initialize alpha and kappa randomly within a specific range?
            Float alpha = 1.f / (Float) numComponents;
            Float kappa = 5.f;

            // initialize mu with spherical Fibonacci point set, which is uniformly distributed on the unit sphere
            // see Marques et al. "Spherical Fibonacci Point Sets for Illumination Integrals" for more details
            Float sinPhi, cosPhi;
            math::sincos(2.f * (Float) k * M_PI * 0.618034f, &sinPhi, &cosPhi);
            Float cosTheta = 1 - (Float) (2 * k + 1) / (Float) numComponents;
            Float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);
            Vector3 mu = {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};

            components.emplace_back(mu, kappa, alpha, 0.f);
        }

        weightSum = 0;
        batchIndex = 0;
    }

    // TODO the sampling procedure may still have numeric issue, fail if encounter nan or inf?
    inline Vector3 sample(Float & pdf, const Point2 & sample) {
        // pick a component
        int k = 0;
        Float accMixWeightSum = 0.f;
        for (; k < components.size() - 1; ++k) {
            if (sample.x < accMixWeightSum + components[k].alpha) {
                break;
            }
            accMixWeightSum += components[k].alpha;
        }

        // sample a direction
        Float x = (sample.x - accMixWeightSum) / components[k].alpha;
        Vector3 vec = sampleComponent(k, {x, sample.y});

        assert(!std::isnan(vec[0]) && !std::isnan(vec[1]) && !std::isnan(vec[2]));

        // one-sample MIS
        // see Veach 1997 "ROBUST MONTE CARLO METHODS FOR LIGHT TRANSPORT SIMULATION"
        pdf = 0;
        for (k = 0; k < components.size(); ++k) {
            pdf += components[k].alpha * pdfComponent(k, vec);
        }

        assert(!std::isnan(pdf));

        return vec;
    }

    inline Float pdf(const Vector3 & vec) {
        if (std::isnan(vec[0] + vec[1] + vec[2])) {
            return 0;
        }

//        assert(!std::isnan(vec[0]) && !std::isnan(vec[1]) && !std::isnan(vec[2]));
        Float pdfValue = 0.f;
        for (int k = 0; k < components.size(); ++k) {
            pdfValue += components[k].alpha * pdfComponent(k, vec);
        }
        assert(!std::isnan(pdfValue));
        return pdfValue;
    }

    // this implementation uses mini-batch stepwise EM
    // see Cappe and Moulines "Online EM Algorithm for Latent Data Models" for more details
    void update(const std::vector<PGSamplingRecord> & samples, int start, int end) {
        auto numComponents = (int) components.size();

        std::vector<Float> lastGammaWeightSums(numComponents);
        std::vector<Vector3> lastGammaWeightSampleSums(numComponents);

        std::vector<Float> partialPdfs(numComponents);
        std::vector<Float> batchGammaWeightSums(numComponents);
        std::vector<Vector3> batchGammaWeightSampleSums(numComponents);

        int batchStart = start;
        while (batchStart < end) {
            int batchEnd = std::min(batchStart + BATCH_SIZE, end);

            Float movingWeight = std::pow(batchIndex + 1, REDUCTION_POWER);

            // compute previous sufficient statistics
            for (int k = 0; k < numComponents; ++k) {
                lastGammaWeightSums[k] = weightSum * components[k].alpha;
                lastGammaWeightSampleSums[k] = components[k].mu * components[k].meanCosine * lastGammaWeightSums[k];
            }

            // compute batch sufficient statistics
            Float batchWeightSum = 0;
            std::fill(batchGammaWeightSums.begin(), batchGammaWeightSums.end(), 0);
            std::fill(batchGammaWeightSampleSums.begin(), batchGammaWeightSampleSums.end(), Vector3(0.f));
            for (int i = batchStart; i < batchEnd; ++i) {
                Float weightI = samples[i].radiance / samples[i].pdf;

                Float pdf = 0.f;
                for (int k = 0; k < numComponents; ++k) {
                    partialPdfs[k] = components[k].alpha * pdfComponent(k, samples[i].direction);
                    pdf += partialPdfs[k];
                }

                // TODO sometimes encounter samples with zero pdf (value is too small)
                pdf = std::max(pdf, std::numeric_limits<Float>::min());

                for (int k = 0; k < numComponents; ++k) {
                    Float gammaIK = partialPdfs[k] / pdf;
                    assert(!std::isnan(gammaIK));
                    batchGammaWeightSums[k] += gammaIK * weightI;
                    batchGammaWeightSampleSums[k] += gammaIK * weightI * samples[i].direction;
                }

                batchWeightSum += weightI;
            }

            // update parameters
            batchIndex += 1;
            weightSum = math::mix(movingWeight, weightSum, batchWeightSum);
            for (int k = 0; k < numComponents; ++k) {
                Float gammaWeightSum = math::mix(movingWeight, lastGammaWeightSums[k], batchGammaWeightSums[k]);
                Vector3 gammaWeightSampleSum = math::mix(movingWeight, lastGammaWeightSampleSums[k], batchGammaWeightSampleSums[k]);

                // TODO run out of precision
                Float rLength = gammaWeightSampleSum.length();
                if (gammaWeightSum == 0 || rLength == 0) {
                    continue;
                }

                components[k].alpha = gammaWeightSum / weightSum;
                components[k].mu = gammaWeightSampleSum / rLength;
                components[k].meanCosine = std::min(rLength / gammaWeightSum, 0.9999f);
                // TODO use a more accurate approximation within the specified range?
                components[k].kappa = math::clamp(components[k].meanCosine *
                    (3.f - components[k].meanCosine * components[k].meanCosine) /
                    (1.f - components[k].meanCosine * components[k].meanCosine), MIN_KAPPA, MAX_KAPPA);
            }

            batchStart = batchEnd;
        }
        // TODO better strategy to handle components with extreme values
    }

//    void update(const std::vector<PGSamplingRecord> & samples, int start, int end) {
//        auto numComponents = (int) components.size();
//
//        // compute previous sufficient statistics
//        std::vector<Float> lastGammaWeightSums(numComponents);
//        std::vector<Vector3> lastGammaWeightSampleSums(numComponents);
//        for (int k = 0; k < numComponents; ++k) {
//            lastGammaWeightSums[k] = weightSum * components[k].alpha;
//            lastGammaWeightSampleSums[k] = components[k].mu * components[k].meanCosine * lastGammaWeightSums[k];
//        }
//
//        // compute batch sample weights
//        Float batchWeightSum = 0.f;
//        std::vector<Float> weights;
//        for (int i = start; i < end; ++i) {
//            weights.push_back(samples[i].radiance / samples[i].pdf);
//            batchWeightSum += weights.back();
//        }
//        weightSum += batchWeightSum;
//
//        std::vector<Float> partialPdfs(numComponents);
//        std::vector<Float> batchGammaWeightSums(numComponents);
//        std::vector<Vector3> batchGammaWeightSampleSums(numComponents);
//
//        // TODO seems to converge slow?
//        const int MAX_ITERATION = 128;
//        const Float LIKELIHOOD_THRESHOLD = 5e-3;
//
//        int iteration = 0;
//        Float lastLogLikelihood = 0;
//        while (iteration < MAX_ITERATION) {
//            Float logLikelihood = 0;
//            std::fill(batchGammaWeightSums.begin(), batchGammaWeightSums.end(), 0);
//            std::fill(batchGammaWeightSampleSums.begin(), batchGammaWeightSampleSums.end(), Vector3(0.f));
//
//            // compute batch sufficient statistics
//            for (int i = start; i < end; ++i) {
//                Float pdfValue = 0.f;
//                for (int k = 0; k < numComponents; ++k) {
//                    assert(!std::isnan(samples[i].direction[0]) && !std::isnan(samples[i].direction[1]) &&
//                           !std::isnan(samples[i].direction[2]));
//                    partialPdfs[k] = components[k].alpha * pdfComponent(k, samples[i].direction);
//                    pdfValue += partialPdfs[k];
//                }
//
//                // TODO sometimes encounter samples with zero pdf (value is too small)
//                pdfValue = std::max(pdfValue, std::numeric_limits<Float>::min());
//
//                for (int k = 0; k < numComponents; ++k) {
//                    Float gammaIK = partialPdfs[k] / pdfValue;
//                    assert(!std::isnan(gammaIK));
//                    assert(!std::isnan(weights[i - start]));
//                    batchGammaWeightSums[k] += gammaIK * weights[i - start];
//                    batchGammaWeightSampleSums[k] += gammaIK * weights[i - start] * samples[i].direction;
//                }
//
//                logLikelihood += weights[i - start] * std::log(pdfValue);
//            }
//
//            // update parameters
//            for (int k = 0; k < numComponents; ++k) {
//                Float gammaWeightSum = lastGammaWeightSums[k] + batchGammaWeightSums[k];
//                Vector3 gammaWeightSampleSum = lastGammaWeightSampleSums[k] + batchGammaWeightSampleSums[k];
//
//                // TODO run out of precision
//                Float rLength = gammaWeightSampleSum.length();
//                if (gammaWeightSum == 0 || rLength == 0) {
//                    continue;
//                }
//
//                components[k].alpha = gammaWeightSum / weightSum;
//                components[k].mu = gammaWeightSampleSum / rLength;
//                components[k].meanCosine = std::min(rLength / gammaWeightSum, 0.9999f);
//                // TODO use a more accurate approximation within the specified range?
//                components[k].kappa = math::clamp(components[k].meanCosine *
//                    (3.f - components[k].meanCosine * components[k].meanCosine) /
//                    (1.f - components[k].meanCosine * components[k].meanCosine), MIN_KAPPA, MAX_KAPPA);
//            }
//
//            if (iteration >= 1) {
//                // TODO the log likelihood can sometimes drop, may be due to the approximation of kappa?
//                if ((logLikelihood - lastLogLikelihood) / std::abs(lastLogLikelihood) < LIKELIHOOD_THRESHOLD) {
//                    break;
//                }
//            }
//
//            iteration += 1;
//            lastLogLikelihood = logLikelihood;
//        }
//
//        if (iteration >= MAX_ITERATION) {
//            std::cout << "Warning: VMM failed to converge within " << MAX_ITERATION << " iterations" << std::endl;
//        }
//
//        // TODO better strategy to handle components with extreme values
//    }

    // TODO better strategy on spatial splitting?
    inline void retarget() {
        weightSum *= 0.25;
    }

private:

    struct VMFData {
        Vector3 mu;
        Float kappa;
        Float alpha;
        Float meanCosine;

        inline VMFData(const Vector3 & mu, Float kappa, Float alpha, Float meanCosine) {
            this->mu = mu;
            this->kappa = kappa;
            this->alpha = alpha;
            this->meanCosine = meanCosine;
        }
    };

    std::vector<VMFData> components;
    Float weightSum;
    Float batchIndex;

    // for vMF sampling and pdf evaluation
    // see Jakob https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    [[nodiscard]]
    inline Vector3 sampleComponent(int k, const Vector2 & rns) const {
        Float sinPhi, cosPhi;
        math::sincos(2 * M_PI * rns.y, &sinPhi, &cosPhi);

        Float cosTheta;
        if (rns.x == 0) {
            cosTheta = -1;
        } else {
            double value = math::mix((double) rns.x, math::fastexp(-2. * components[k].kappa), 1.);
            cosTheta = (Float) math::clamp(1. + math::fastlog(value) / components[k].kappa, -1., 1.);
        }
        Float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        return Frame(components[k].mu).toWorld({
            sinTheta * cosPhi,
            sinTheta * sinPhi,
            cosTheta
        });
    }

    [[nodiscard]]
    inline Float pdfComponent(int k, const Vector3 & vec) const {
//        assert(!std::isnan(vec[0]) && !std::isnan(vec[1]) && !std::isnan(vec[2]));
//        assert(!std::isnan(components[k].mu[0]) && !std::isnan(components[k].mu[1]) && !std::isnan(components[k].mu[2]));
        Float numerator = components[k].kappa * math::fastexp(components[k].kappa * (dot(components[k].mu, vec) - 1.f));
//        assert(!std::isnan(numerator));
        Float denominator = (2.f * M_PI * (1.f - math::fastexp(-2.f * components[k].kappa)));
//        assert(!std::isnan(denominator));
//        assert(denominator != 0);
        Float pdfValue = numerator / denominator;
//        assert(!std::isnan(pdfValue));
        return pdfValue;
    }
};

MTS_NAMESPACE_END
