#pragma once
#include <vector>

MTS_NAMESPACE_BEGIN

// TODO insufficient valid samples on the scene "clocks"
class VMM {
public:

    // TODO expand the value range of kappa
    constexpr static Float MIN_KAPPA = 1e-2;
    constexpr static Float MAX_KAPPA = 1e+4;
    constexpr static Float REDUCTION_POWER = -0.65;
    constexpr static int BATCH_SIZE = 64;
    constexpr static int NUM_INITIAL_COMPONENTS = 20;

    inline explicit VMM(int numComponents = NUM_INITIAL_COMPONENTS) {
        alpha.reserve(numComponents);
        kappa.reserve(numComponents);
        mu.reserve(numComponents);

        for (int k = 0; k < numComponents; ++k) {
            // TODO set alpha and kappa randomly within a specific range?
            alpha.emplace_back(1.f / numComponents);
            kappa.emplace_back(5);
            // initialize mu with spherical Fibonacci point set, which is uniformly distributed on the unit sphere
            // see Marques et al. "Spherical Fibonacci Point Sets for Illumination Integrals" for more details
            Float sinPhi, cosPhi;
            math::sincos(2.f * (Float) k * M_PI * 0.618034f, &sinPhi, &cosPhi);
            Float cosTheta = 1 - (Float) (2 * k + 1) / (Float) numComponents;
            Float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);
            mu.emplace_back(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        }

        r.assign(numComponents, 0);
        distance.assign(numComponents, std::numeric_limits<Float>::infinity());
        distanceWeightSum.assign(numComponents, 0);

        center = Vector3(0, 0, 0);
        weightSum = 0;
        batchIndex = 0;
        firstUpdate = true;
    }

    // TODO the sampling procedure may still have numeric issue, fail if encounter nan or inf?
    inline Vector3 sample(const Point2 & rn, Float & pdf, const Vector3 & position) const {
        size_t numComponents = alpha.size();

        // warp to the requested position
        Vector3 warpedMu[numComponents];
        for (size_t k = 0; k < numComponents; ++k) {
            Vector3 originK = center + mu[k] * distance[k];
            warpedMu[k] = normalize(originK - position);
            if (std::isnan(warpedMu[k][0]) || std::isnan(warpedMu[k][1]) || std::isnan(warpedMu[k][2])) {
//                std::cout << "center = " << center.toString() << ", "
//                          << "mu[k] = " << mu[k].toString() << ", "
//                          << "distance[k] = " << distance[k] << ", "
//                          << "originK = " << originK.toString() << ", "
//                          << "position = " << position.toString() << ", "
//                          << "warpedMu[k] = " << warpedMu[k].toString()
//                          << std::endl;
//                assert(false);

                std::cout << "originK == position!" << std::endl;
                warpedMu[k] = mu[k];
            }
        }

        // pick a component
        size_t k = 0;
        Float accMixWeightSum = 0.f;
        for (; k < alpha.size() - 1; ++k) {
            if (rn.x < accMixWeightSum + alpha[k]) {
                break;
            }
            accMixWeightSum += alpha[k];
        }

        // sample a direction
        Float x = (rn.x - accMixWeightSum) / alpha[k];
        Vector3 omega = sampleComponent(warpedMu[k], kappa[k], {x, rn.y});
//        Vector3 omega = sampleComponent(mu[k], kappa[k], {x, rn.y});

        // one-sample MIS
        // see Veach 1997 "ROBUST MONTE CARLO METHODS FOR LIGHT TRANSPORT SIMULATION"
        pdf = 0;
        for (k = 0; k < numComponents; ++k) {
            pdf += alpha[k] * pdfComponent(warpedMu[k], kappa[k], omega);
//            pdf += alpha[k] * pdfComponent(mu[k], kappa[k], omega);
        }
        assert(!std::isnan(pdf));

        return omega;
    }

    inline Float pdf(const Vector3 & omega, const Vector3 & position) const {
        size_t numComponents = alpha.size();

        // warp to the requested position
        Vector3 warpedMu[numComponents];
        for (size_t k = 0; k < numComponents; ++k) {
            Vector3 originK = center + mu[k] * distance[k];
            warpedMu[k] = normalize(originK - position);
            if (std::isnan(warpedMu[k][0]) || std::isnan(warpedMu[k][1]) || std::isnan(warpedMu[k][2])) {
//                std::cout << "center = " << center.toString() << ", "
//                          << "mu[k] = " << mu[k].toString() << ", "
//                          << "distance[k] = " << distance[k] << ", "
//                          << "originK = " << originK.toString() << ", "
//                          << "position = " << position.toString()
//                          << std::endl;
//                assert(false);

                std::cout << "originK == position!" << std::endl;
                warpedMu[k] = mu[k];
            }
        }

        Float value = 0.f;
        for (size_t k = 0; k < alpha.size(); ++k) {
            value += alpha[k] * pdfComponent(warpedMu[k], kappa[k], omega);
//            value += alpha[k] * pdfComponent(mu[k], kappa[k], omega);
        }
        assert(!std::isnan(value));
        return value;
    }

    void update(std::vector<PGSamplingRecord> & samples, int start, int end, const Vector3 & position) {
        size_t numComponents = alpha.size();

        if (!firstUpdate) {
            // warp to the new position
            for (size_t k = 0; k < numComponents; ++k) {
                Vector3 originK = center + mu[k] * distance[k];
                mu[k] = originK - position;
                distance[k] = mu[k].length();
                assert(distance[k] > 0);
                mu[k] /= distance[k];
            }
        } else {
            firstUpdate = false;
        }

        center = position;
//
//        // relocate the samples
        for (int i = start; i < end; ++i) {
            Vector3 originI = samples[i].position + samples[i].direction * samples[i].distance;
            samples[i].direction = originI - position;
            samples[i].distance = samples[i].direction.length();
            samples[i].direction /= samples[i].distance;
            assert(samples[i].distance > 0);
        }

        updateComponents(samples, start, end);

        // update the perceived distance
        std::vector<Float> componentPdfs(numComponents, 0);
        std::vector<Float> batchWeightedDistanceSum(numComponents, 0);
        std::vector<Float> batchDistanceWeightSum(numComponents, 0);
        for (int i = start; i < end; ++i) {
            assert(samples[i].distance > 0);
            assert(!std::isnan(samples[i].distance));

            Float weightI = samples[i].radiance / samples[i].pdf;
            Float pdf = 0;
            for (size_t k = 0; k < numComponents; ++k) {
                componentPdfs[k] = pdfComponent(mu[k], kappa[k], samples[i].direction);
                pdf += alpha[k] * componentPdfs[k];
            }

            for (size_t k = 0; k < numComponents; ++k) {
                Float gammaIK = alpha[k] * componentPdfs[k] / pdf;
                if (std::isnan(gammaIK)) {
                    std::cout << "alpha[k] = " << alpha[k] << ", "
                              << "componentPdfs[k] = " << componentPdfs[k] << ", "
                              << "pdf = " << pdf
                              << std::endl;
                    assert(false);
                }

                Float distanceWeight = weightI * gammaIK * componentPdfs[k];
                assert(!std::isnan(distanceWeight));

                batchWeightedDistanceSum[k] += distanceWeight / samples[i].distance;
                batchDistanceWeightSum[k] += distanceWeight;

//                if (distanceWeight == 0) {
//                    std::cout << "partialPdf = " << partialPdfs[k] << ", "
//                              << "gammaIK = " << gammaIK << ", "
//                              << "weightI = " << weightI
//                              << std::endl;
//                    assert(0);
//                }
            }
        }
        for (size_t k = 0; k < numComponents; ++k) {
            assert(!std::isnan(batchDistanceWeightSum[k]));
            assert(!std::isnan(batchWeightedDistanceSum[k]));
//            if (weightedDistanceSum[k] <= 0) {
//                std::cout << "weightedDistanceSum = " << weightedDistanceSum[k] << ", "
//                          << "distanceWeightSum = " << distanceWeightSum[k]
//                          << std::endl;
//                assert(false);
//            }
            if (batchDistanceWeightSum[k] == 0 || batchWeightedDistanceSum[k] == 0) {
                continue;
            }

            Float weightedDistanceSum = distanceWeightSum[k] / distance[k];
            distanceWeightSum[k] += batchDistanceWeightSum[k];
            distance[k] = distanceWeightSum[k] / (weightedDistanceSum + batchWeightedDistanceSum[k]);
//            if (std::isnan(distance[k]) || distance[k] <= 0) {
//                std::cout << ""
//            }
        }
    }

    // TODO better strategy on spatial splitting?
    inline void retarget() {
        weightSum *= 0.25;
////        std::fill(distanceWeightSum.begin(), distanceWeightSum.end(), 0);
        for (size_t k = 0; k < alpha.size(); ++k) {
            distanceWeightSum[k] *= 0.25;
        }
//        weightSum = 0;
//        std::fill(distanceWeightSum.begin(), distanceWeightSum.end(), std::numeric_limits<Float>::infinity());
    }

    std::vector<Vector3> mu;
    std::vector<Float> kappa;
    std::vector<Float> alpha;
    std::vector<Float> r;
    std::vector<Float> distance;
    std::vector<Float> distanceWeightSum;

    Vector3 center;
    Float weightSum;
    Float batchIndex;
    bool firstUpdate;

    // for vMF sampling and pdf evaluation
    // see Jakob https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    // TODO check precision requirements
    inline static Vector3 sampleComponent(const Vector3 & muK, Float kappaK, const Vector2 & rn) {
        Float sinPhi, cosPhi;
        math::sincos(2 * M_PI * rn.y, &sinPhi, &cosPhi);

        Float cosTheta;
        if (rn.x == 0) {
            cosTheta = -1;
        } else {
            double value = math::mix((double) rn.x, math::fastexp(-2. * kappaK), 1.);
            cosTheta = (Float) math::clamp(1. + math::fastlog(value) / kappaK, -1., 1.);
        }
        Float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        return Frame(muK).toWorld({
            sinTheta * cosPhi,
            sinTheta * sinPhi,
            cosTheta
        });
    }

    inline static Float pdfComponent(const Vector3 & muK, Float kappaK, const Vector3 & omega) {
        if (std::isnan(omega[0]) || std::isnan(omega[1]) || std::isnan(omega[2])) {
            return 0;
        }
        Float nu = kappaK * math::fastexp(kappaK * (dot(muK, omega) - 1.f));
        Float de = (2.f * M_PI * (1.f - math::fastexp(-2.f * kappaK)));
        assert(!std::isnan(nu));
        assert(!std::isnan(de));
        assert(de != 0);
        Float value = nu / de;
        if (std::isnan(value)) {
            std::cout << "mu = " << muK.toString() << ", "
                      << "kappa = " << kappaK << ", "
                      << "omega = " << omega.toString()
                      << std::endl;
            assert(false);
        }
        return value;
    }

    // this implementation uses mini-batch stepwise EM
    // see Cappe and Moulines "Online EM Algorithm for Latent Data Models" for more details
//    void updateComponents(const std::vector<PGSamplingRecord> & samples, int start, int end) {
//        size_t numComponents = alpha.size();
//
//        // compute previous sufficient statistics
//        Float gammaWeightSum[numComponents];
//        Vector3 gammaWeightSampleSum[numComponents];
//        for (int k = 0; k < numComponents; ++k) {
//            gammaWeightSum[k] = weightSum * alpha[k];
//            gammaWeightSampleSum[k] = mu[k] * r[k] * gammaWeightSum[k];
//        }
//
//        Float partialPdf[numComponents];
//        Float batchGammaWeightSum[numComponents];
//        Vector3 batchGammaWeightSampleSum[numComponents];
//
//        int batchStart = start;
//        while (batchStart < end) {
//            int batchEnd = std::min(batchStart + BATCH_SIZE, end);
//            Float movingWeight = std::pow(batchIndex + 1, REDUCTION_POWER);
//
//            // compute batch sufficient statistics
//            Float batchWeightSum = 0;
//            std::fill(batchGammaWeightSum, batchGammaWeightSum + numComponents, 0);
//            std::fill(batchGammaWeightSampleSum, batchGammaWeightSampleSum + numComponents, Vector3(0.f));
//            for (int i = batchStart; i < batchEnd; ++i) {
//                Float weightI = samples[i].radiance / samples[i].pdf;
//
//                Float pdf = 0.f;
//                for (int k = 0; k < numComponents; ++k) {
//                    partialPdf[k] = alpha[k] * pdfComponent(mu[k], kappa[k], samples[i].direction);
//                    pdf += partialPdf[k];
//                }
//
//                // TODO sometimes encounter samples with zero pdf (value is too small)
//                pdf = std::max(pdf, std::numeric_limits<Float>::min());
//
//                for (int k = 0; k < numComponents; ++k) {
//                    Float gammaIK = partialPdf[k] / pdf;
//                    assert(!std::isnan(gammaIK));
//                    batchGammaWeightSum[k] += gammaIK * weightI;
//                    batchGammaWeightSampleSum[k] += gammaIK * weightI * samples[i].direction;
//                }
//
//                batchWeightSum += weightI;
//            }
//
//            // update parameters
//            batchIndex += 1;
//            weightSum = math::mix(movingWeight, weightSum, batchWeightSum);
//            for (int k = 0; k < numComponents; ++k) {
//                gammaWeightSum[k] = math::mix(movingWeight, gammaWeightSum[k], batchGammaWeightSum[k]);
//                gammaWeightSampleSum[k] = math::mix(movingWeight, gammaWeightSampleSum[k], batchGammaWeightSampleSum[k]);
//
//                // TODO run out of precision
//                Float rLength = gammaWeightSampleSum[k].length();
//                if (gammaWeightSum[k] == 0 || rLength == 0) {
//                    continue;
//                }
//
//                alpha[k] = gammaWeightSum[k] / weightSum;
//                mu[k] = gammaWeightSampleSum[k] / rLength;
//                r[k] = std::min(rLength / gammaWeightSum[k], 0.9999f);
//                // TODO use a more accurate approximation within the specified range?
//                kappa[k] = math::clamp(r[k] * (3.f - r[k] * r[k]) / (1.f - r[k] * r[k]), MIN_KAPPA, MAX_KAPPA);
//            }
//
//            batchStart = batchEnd;
//        }
//        // TODO better strategy to handle components with extreme values
//    }

    void updateComponents(const std::vector<PGSamplingRecord> & samples, int start, int end) {
        size_t numComponents = alpha.size();

        // compute previous sufficient statistics
        std::vector<Float> lastGammaWeightSums(numComponents);
        std::vector<Vector3> lastGammaWeightSampleSums(numComponents);
        for (int k = 0; k < numComponents; ++k) {
            lastGammaWeightSums[k] = weightSum * alpha[k];
            lastGammaWeightSampleSums[k] = mu[k] * r[k] * lastGammaWeightSums[k];
        }

        // compute batch sample weights
        Float batchWeightSum = 0.f;
        std::vector<Float> weights;
        for (int i = start; i < end; ++i) {
            weights.push_back(samples[i].radiance / samples[i].pdf);
            batchWeightSum += weights.back();
        }
        weightSum += batchWeightSum;

        std::vector<Float> partialPdfs(numComponents);
        std::vector<Float> batchGammaWeightSums(numComponents);
        std::vector<Vector3> batchGammaWeightSampleSums(numComponents);

        // TODO seems to converge slow?
        const int MAX_ITERATION = 128;
        const Float LIKELIHOOD_THRESHOLD = 5e-3;

        int iteration = 0;
        Float lastLogLikelihood = 0;
        while (iteration < MAX_ITERATION) {
            Float logLikelihood = 0;
            std::fill(batchGammaWeightSums.begin(), batchGammaWeightSums.end(), 0);
            std::fill(batchGammaWeightSampleSums.begin(), batchGammaWeightSampleSums.end(), Vector3(0.f));

            // compute batch sufficient statistics
            for (int i = start; i < end; ++i) {
                Float pdfValue = 0.f;
                for (int k = 0; k < numComponents; ++k) {
                    assert(!std::isnan(samples[i].direction[0]) && !std::isnan(samples[i].direction[1]) &&
                           !std::isnan(samples[i].direction[2]));
                    partialPdfs[k] = alpha[k] * pdfComponent(mu[k], kappa[k], samples[i].direction);
                    pdfValue += partialPdfs[k];
                }

                // TODO sometimes encounter samples with zero pdf (value is too small)
                pdfValue = std::max(pdfValue, std::numeric_limits<Float>::min());

                for (int k = 0; k < numComponents; ++k) {
                    Float gammaIK = partialPdfs[k] / pdfValue;
                    assert(!std::isnan(gammaIK));
                    assert(!std::isnan(weights[i - start]));
                    batchGammaWeightSums[k] += gammaIK * weights[i - start];
                    batchGammaWeightSampleSums[k] += gammaIK * weights[i - start] * samples[i].direction;
                }

                logLikelihood += weights[i - start] * std::log(pdfValue);
            }

            // update parameters
            for (int k = 0; k < numComponents; ++k) {
                Float gammaWeightSum = lastGammaWeightSums[k] + batchGammaWeightSums[k];
                Vector3 gammaWeightSampleSum = lastGammaWeightSampleSums[k] + batchGammaWeightSampleSums[k];

                // TODO run out of precision
                Float rLength = gammaWeightSampleSum.length();
                if (gammaWeightSum == 0 || rLength == 0) {
                    continue;
                }

                alpha[k] = gammaWeightSum / weightSum;
                mu[k] = gammaWeightSampleSum / rLength;
                r[k] = std::min(rLength / gammaWeightSum, 0.9999f);
                // TODO use a more accurate approximation within the specified range?
                kappa[k] = math::clamp(r[k] * (3.f - r[k] * r[k]) / (1.f - r[k] * r[k]), MIN_KAPPA, MAX_KAPPA);
            }

            if (iteration >= 1) {
                // TODO the log likelihood can sometimes drop, may be due to the approximation of kappa?
                if ((logLikelihood - lastLogLikelihood) / std::abs(lastLogLikelihood) < LIKELIHOOD_THRESHOLD) {
                    break;
                }
            }

            iteration += 1;
            lastLogLikelihood = logLikelihood;
        }

        if (iteration >= MAX_ITERATION) {
            std::cout << "Warning: VMM failed to converge within " << MAX_ITERATION << " iterations" << std::endl;
        }

        // TODO better strategy to handle components with extreme values
    }
};

MTS_NAMESPACE_END
