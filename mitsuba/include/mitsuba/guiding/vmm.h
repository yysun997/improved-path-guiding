#pragma once
#include <numeric>

#define VCL_NAMESPACE vcl
#include "vcl/vectorclass.h"
#include "vcl/vectormath_exp.h"

MTS_NAMESPACE_BEGIN

namespace PathGuiding::vmm {

using Scalar = float;
using SampleIterator = std::vector<SampleData>::iterator;

#if INSTRSET >= 8
    using Packet = vcl::Vec8f;
    using Mask = vcl::Vec8fb;
#elif INSTRSET >= 2
    using Packet = vcl::Vec4f;
    using Mask = vcl::Vec4fb;
#endif

using PacketVec3 = TVector3<Packet>;

constexpr int NComponents = 32;
constexpr int NScalars = Packet::size();
constexpr int NKernels = (NComponents + NScalars - 1) / NScalars;

// TODO approximate the exp?

// for vMF sampling and pdf evaluation
// see Jakob https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
class VMFKernel {
public:

    const PacketVec3 mu;
    const Packet kappa{};
    const Packet alpha{};

    inline VMFKernel & operator=(const VMFKernel & kernel) {
        const_cast<PacketVec3 &>(mu) = kernel.mu;
        const_cast<Packet &>(kappa) = kernel.kappa;
        const_cast<Packet &>(alpha) = kernel.alpha;
        eMin2Kappa = kernel.eMin2Kappa;
        pdfFactor = kernel.pdfFactor;
        return *this;
    }

    inline void setMu(int i, const Vector3 & v) {
        const_cast<PacketVec3 &>(mu)[0].insert(i, v[0]);
        const_cast<PacketVec3 &>(mu)[1].insert(i, v[1]);
        const_cast<PacketVec3 &>(mu)[2].insert(i, v[2]);
    }

    inline void setMu(const PacketVec3 & pv, const Mask & m) {
        const_cast<PacketVec3 &>(mu)[0] = vcl::select(m, pv[0], mu[0]);
        const_cast<PacketVec3 &>(mu)[1] = vcl::select(m, pv[1], mu[1]);
        const_cast<PacketVec3 &>(mu)[2] = vcl::select(m, pv[2], mu[2]);
    }

    inline void setKappa(const Packet & k) {
        const_cast<Packet &>(kappa) = k;
        refreshCache();
    }

    inline void setKappa(const Packet & k, const Mask & m) {
        const_cast<Packet &>(kappa) = vcl::select(m, k, kappa);
        refreshCache();
    }

    inline void setAlpha(const Packet & a) {
        const_cast<Packet &>(alpha) = a;
    }

    inline void setAlpha(const Packet & a, const Mask & m) {
        const_cast<Packet &>(alpha) = vcl::select(m, a, alpha);
    }

    [[nodiscard]]
    inline Vector3 sample(int i, const Vector2 & rn) const {
        Vector3 muI(mu[0][i], mu[1][i], mu[2][i]);
        Scalar kappaI = kappa[i];

        Scalar sinPhi, cosPhi;
        math::sincos(2 * M_PI * rn.y, &sinPhi, &cosPhi);

        Scalar value = rn.x + (1.f - rn.x) * eMin2Kappa[i];
        Scalar cosTheta = math::clamp(1.f + std::log(value) / kappaI, -1.f, 1.f);
        Scalar sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        return Frame(muI).toWorld({
            sinTheta * cosPhi, sinTheta * sinPhi, cosTheta
        });
    }

    [[nodiscard]]
    inline Packet pdf(const Vector3 & v) const {
        Packet t = dot(mu, PacketVec3(v)) - 1.f;
        Packet e = vcl::exp(kappa * t);
        return pdfFactor * e;
    }

private:

    friend class VMFMixture;
    friend class ParallaxAwareVMM;

    Packet eMin2Kappa{};
    Packet pdfFactor{};

    inline VMFKernel() = default;

    inline void refreshCache() {
        eMin2Kappa = vcl::exp(-2.f * kappa);
        Packet de = 2 * M_PI * (1.f - eMin2Kappa);
        pdfFactor = kappa / de;
    }
};

class VMFMixture {
public:

    [[nodiscard]]
    inline Vector3 sample(const Vector2 & rn) const {
        // pick a kernel
        int k = 0;
        Scalar accAlphaSum = 0;
        for (; k < NKernels - 1; ++k) {
            Scalar kernelAlphaSum = vcl::horizontal_add(kernels[k].alpha);
            if (rn.x < accAlphaSum + kernelAlphaSum) {
                break;
            }
            accAlphaSum += kernelAlphaSum;
        }

        // pick a component
        int i = 0;
        for (; i < NScalars - 1; ++i) {
            if (rn.x < accAlphaSum + kernels[k].alpha[i]) {
                break;
            }
            accAlphaSum += kernels[k].alpha[i];
        }
        Vector2 reused((rn.x - accAlphaSum) / kernels[k].alpha[i], rn.y);

        return kernels[k].sample(i, reused);
    }

    [[nodiscard]]
    inline Scalar pdf(const Vector3 & w) const {
        Packet value(0);

        for (const auto & kernel: kernels) {
            value += kernel.alpha * kernel.pdf(w);
        }

        return vcl::horizontal_add(value);
    }

private:

    friend class ParallaxAwareVMM;

    VMFKernel kernels[NKernels];

};

class ParallaxAwareVMM : public VMFMixture {
public:

    inline ParallaxAwareVMM() {
        for (int c = 0; c < NComponents; ++c) {
            // initialize mu with spherical Fibonacci point set, which is uniformly distributed on the unit sphere
            // see Marques et al. "Spherical Fibonacci Point Sets for Illumination Integrals" for more details
            Scalar sinPhi, cosPhi;
            math::sincos(2.f * (Scalar) c * M_PI * 0.618034f, &sinPhi, &cosPhi);

            Scalar cosTheta = 1 - (Scalar) (2 * c + 1) / NComponents;
            Scalar sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

            Vector3 mu(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            kernels[c / NScalars].setMu(c % NScalars, mu);
        }

        for (int k = 0; k < NKernels; ++k) {
            kernels[k].setAlpha(1.f / NComponents);
            kernels[k].setKappa(5);
            meanCosine[k] = 0;
            distances[k] = std::numeric_limits<Scalar>::infinity();
            distanceWeightSums[k] = 0;
        }

        currentPosition = Vector3(0.f);
        sampleWeightSum = 0;
        batchIndex = 0;
    }

    inline void update(const Vector3 & newPosition, SampleIterator begin, SampleIterator end) {
        warpTo(newPosition);

        // relocate the samples
        for (auto iter = begin; iter != end; ++iter) {
            Vector3 origin = iter->position + iter->direction * iter->distance;
            Vector3 po = origin - newPosition;
            Scalar t = po.length();
            if (t > 0) {
                iter->direction = po / t;
                iter->distance = t;
            }
        }

        updateKernels(begin, end);
        updateDistances(begin, end);
    }

    inline void getWarped(VMFMixture & out, const Vector3 & newPosition) const {
        PacketVec3 newToCurrent(currentPosition - newPosition);
        for (int k = 0; k < NKernels; ++k) {
            PacketVec3 po = kernels[k].mu * distances[k] + newToCurrent;
            Packet t = vcl::sqrt(dot(po, po));
            Mask mask = vcl::is_finite(distances[k]) && t > 0;

            out.kernels[k] = kernels[k];
            out.kernels[k].setMu(po / t, mask);
        }
    }

    inline ParallaxAwareVMM * split() {
        sampleWeightSum *= 0.25;
        for (auto & distanceWeightSum : distanceWeightSums) {
            distanceWeightSum *= 0.25f;
        }

        return new ParallaxAwareVMM(*this);
    }

private:

    Packet meanCosine[NKernels]{};
    Packet distances[NKernels]{};
    Packet distanceWeightSums[NKernels]{};
    Vector3 currentPosition;
    Scalar sampleWeightSum{};
    Scalar batchIndex{};

    inline static Scalar mix(Scalar t, Scalar a, Scalar b) {
        return (1.f - t) * a + t * b;
    }

    inline static Packet mix(Scalar t, const Packet & p1, const Packet & p2) {
        return (1.f - t) * p1 + t * p2;
    }

    inline static PacketVec3 mix(Scalar t, const PacketVec3 & pv1, const PacketVec3 & pv2) {
        return Packet(1.f - t) * pv1 + Packet(t) * pv2;
    }

    // TODO use a more accurate approximation within the specified range?
    inline static Packet meanCosineToKappa(const Packet & meanCosine) {
        Packet kappa = meanCosine * (3.f - meanCosine * meanCosine) / (1.f - meanCosine * meanCosine);
        return vcl::max(vcl::min(kappa, 1e+4f), 1e-2f);
    }

    inline void warpTo(const Vector3 & newPosition) {
        PacketVec3 newToCurrent(currentPosition - newPosition);
        for (int k = 0; k < NKernels; ++k) {
            PacketVec3 po = kernels[k].mu * distances[k] + newToCurrent;
            Packet t = vcl::sqrt(dot(po, po));
            Mask mask = vcl::is_finite(distances[k]) && t > 0;
            // update distance, update mu
            distances[k] = vcl::select(mask, t, distances[k]);
            kernels[k].setMu(po / t, mask);
        }
        currentPosition = newPosition;
    }

    void updateKernels(SampleIterator begin, SampleIterator end) {
        // compute previous sufficient statistics
        Packet lastGammaWeightSums[NKernels];
        PacketVec3 lastGammaWeightSampleSums[NKernels];
        for (int k = 0; k < NKernels; ++k) {
            lastGammaWeightSums[k] = sampleWeightSum * kernels[k].alpha;
            lastGammaWeightSampleSums[k] = kernels[k].mu * meanCosine[k] * lastGammaWeightSums[k];
        }

        // compute sample weights
        Scalar batchSampleWeightSum = std::accumulate(begin, end, 0.f,
            [](Scalar sum, const SampleData & sample) -> Scalar {
                return sum + sample.radiance / sample.pdf;
            }
        );

        batchIndex += 1;
        Scalar movingWeight = 1.f / batchIndex;
        sampleWeightSum = mix(movingWeight, sampleWeightSum, batchSampleWeightSum);

        Packet partialPdfs[NKernels];
        Packet batchGammaWeightSums[NKernels];
        PacketVec3 batchGammaWeightSampleSums[NKernels];

        const int maxIterations = 128;
        const Scalar threshold = 5e-3;

        int iteration = 0;
        Scalar lastLogLikelihood = 0;
        while (iteration < maxIterations) {
            Scalar logLikelihood = 0;
            std::fill(batchGammaWeightSums, batchGammaWeightSums + NKernels, Packet(0));
            std::fill(batchGammaWeightSampleSums, batchGammaWeightSampleSums + NKernels, PacketVec3(0.f));

            // compute batch sufficient statistics
            int i = 0;
            for (auto iter = begin; iter != end; ++iter) {
                Scalar pdf = 0.f;
                for (int k = 0; k < NKernels; ++k) {
                    partialPdfs[k] = kernels[k].alpha * kernels[k].pdf(iter->direction);
                    pdf += vcl::horizontal_add(partialPdfs[k]);
                }

                // TODO sometimes encounter samples with zero pdf (value is too small)
                pdf = std::max(pdf, std::numeric_limits<Scalar>::min());

                Scalar weight = iter->radiance / iter->pdf;
                for (int k = 0; k < NKernels; ++k) {
                    Packet gammaIK = partialPdfs[k] / pdf;
                    batchGammaWeightSums[k] += gammaIK * weight;
                    batchGammaWeightSampleSums[k] += gammaIK * weight * PacketVec3(iter->direction);
                }

                logLikelihood += weight * std::log(pdf);
                i += 1;
            }

            // update parameters
            for (int k = 0; k < NKernels; ++k) {
                Packet gammaWeightSum = mix(movingWeight, lastGammaWeightSums[k], batchGammaWeightSums[k]);
                PacketVec3 gammaWeightSampleSum = mix(movingWeight, lastGammaWeightSampleSums[k], batchGammaWeightSampleSums[k]);
                Packet rLength = vcl::sqrt(dot(gammaWeightSampleSum, gammaWeightSampleSum));

                Mask mask = gammaWeightSum > 0 && rLength > 0;

                kernels[k].setAlpha(gammaWeightSum / sampleWeightSum, mask);
                kernels[k].setMu(gammaWeightSampleSum / rLength, mask);
                meanCosine[k] = vcl::select(mask, vcl::min(rLength / gammaWeightSum, 0.9999f), meanCosine[k]);
                kernels[k].setKappa(meanCosineToKappa(meanCosine[k]), mask);
            }

            if (iteration >= 1) {
                // TODO the log likelihood can sometimes drop, may be due to the approximation of kappa?
                if ((logLikelihood - lastLogLikelihood) / std::abs(lastLogLikelihood) < threshold) {
                    break;
                }
            }

            iteration += 1;
            lastLogLikelihood = logLikelihood;
        }
        // TODO better strategy to handle components with extreme values?
    }

    void updateDistances(SampleIterator begin, SampleIterator end) {
        // update the perceived distances
        Packet componentPdfs[NKernels];
        Packet batchWeightedDistanceSums[NKernels];
        Packet batchDistanceWeightSums[NKernels];

        std::fill(batchWeightedDistanceSums, batchWeightedDistanceSums + NKernels, Packet(0.f));
        std::fill(batchDistanceWeightSums, batchDistanceWeightSums + NKernels, Packet(0.f));

        for (auto iter = begin; iter != end; ++iter) {
            Scalar weight = iter->radiance / iter->pdf;

            Scalar pdf = 0;
            for (int k = 0; k < NKernels; ++k) {
                componentPdfs[k] = kernels[k].pdf(iter->direction);
                pdf += vcl::horizontal_add(kernels[k].alpha * componentPdfs[k]);
            }

            for (int k = 0; k < NKernels; ++k) {
                Packet gammaIK = kernels[k].alpha * componentPdfs[k] / pdf;
                Packet distanceWeight = weight * gammaIK * componentPdfs[k];
                batchWeightedDistanceSums[k] += distanceWeight / iter->distance;
                batchDistanceWeightSums[k] += distanceWeight;
            }
        }

        Scalar movingWeight = 1.f / batchIndex;
        for (int k = 0; k < NKernels; ++k) {
            Mask mask = batchDistanceWeightSums[k] > 0 && batchWeightedDistanceSums[k] > 0;
            Packet weightedDistanceSum = distanceWeightSums[k] / distances[k];
            Packet newWeightSum = mix(movingWeight, distanceWeightSums[k], batchDistanceWeightSums[k]);
            Packet newWeightedDistanceSum = mix(movingWeight, weightedDistanceSum, batchWeightedDistanceSums[k]);
            distances[k] = vcl::select(mask, newWeightSum / newWeightedDistanceSum, distances[k]);
            distanceWeightSums[k] = vcl::select(mask, newWeightSum, distanceWeightSums[k]);
        }
    }
};

}

MTS_NAMESPACE_END
