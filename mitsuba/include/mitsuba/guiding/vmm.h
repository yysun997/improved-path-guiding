#pragma once
#include <numeric>
#include <experimental/simd>

MTS_NAMESPACE_BEGIN

// TODO 10 times slower than robust vmm?
namespace PathGuiding::vmm {

using Scalar = Float;
using Packet = std::experimental::native_simd<Scalar>;
using Mask = std::experimental::native_simd_mask<Scalar>;
using PacketVec3 = TVector3<Packet>;
using SampleIterator = std::vector<SampleData>::iterator;

constexpr size_t NComponents = 32;
constexpr size_t NScalars = Packet::size();
constexpr size_t NKernels = (NComponents + NScalars - 1) / NScalars;

// for vMF sampling and pdf evaluation
// see Jakob https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
class VMFKernel {
private:

    friend class VMFMixture;
    friend class ParallaxAwareVMM;

    PacketVec3 mu;
    Packet kappa{};
    Packet alpha{};

    inline VMFKernel() = default;

    [[nodiscard]]
    inline Vector3 sample(size_t i, const Vector2 & rn) const {
        Vector3 muI(mu[0][i], mu[1][i], mu[2][i]);
        Scalar kappaI = kappa[i];

        Scalar sinPhi, cosPhi;
        math::sincos(2 * M_PI * rn.y, &sinPhi, &cosPhi);

        Scalar value = rn.x + (1.f - rn.x) * math::fastexp(-2 * kappaI);
        Scalar cosTheta = math::clamp(1.f + math::fastlog(value) / kappaI, -1.f, 1.f);
        Scalar sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        return Frame(muI).toWorld({
            sinTheta * cosPhi, sinTheta * sinPhi, cosTheta
        });
    }

    [[nodiscard]]
    inline Packet pdf(const Vector3 & w) const {
        PacketVec3 packetW(w);
        Packet nu = kappa * std::experimental::exp(kappa * (dot(mu, packetW) - Packet(1)));
        Packet de = (2.f * M_PI * (1.f - std::experimental::exp(-2.f * kappa)));
        return nu / de;
    }
};

class VMFMixture {
public:

    [[nodiscard]]
    inline Vector3 sample(const Vector2 & rn) const {
        // pick a kernel
        size_t k = 0;
        Scalar accAlphaSum = 0;
        for (; k < NKernels - 1; ++k) {
            Scalar kernelAlphaSum = std::experimental::reduce(kernels[k].alpha, std::plus<>());
            if (rn.x < accAlphaSum + kernelAlphaSum) {
                break;
            }
            accAlphaSum += kernelAlphaSum;
        }

        // pick a component
        size_t i = 0;
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

        return std::experimental::reduce(value, std::plus<>());
    }

private:

    friend class ParallaxAwareVMM;

    VMFKernel kernels[NKernels];

};

class ParallaxAwareVMM : public VMFMixture {
public:

    inline ParallaxAwareVMM() {
        for (size_t c = 0; c < NComponents; ++c) {
            // initialize mu with spherical Fibonacci point set, which is uniformly distributed on the unit sphere
            // see Marques et al. "Spherical Fibonacci Point Sets for Illumination Integrals" for more details
            Scalar sinPhi, cosPhi;
            math::sincos(2.f * (Scalar) c * M_PI * 0.618034f, &sinPhi, &cosPhi);

            Scalar cosTheta = 1 - (Scalar) (2 * c + 1) / NComponents;
            Scalar sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

            Vector3 mu(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            set(kernels[c / NScalars].mu, c % NScalars, mu);
        }

        for (size_t k = 0; k < NKernels; ++k) {
            kernels[k].alpha = 1.f / NComponents;
            kernels[k].kappa = 5;
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
        PacketVec3 packetCurPos(currentPosition);
        PacketVec3 packetNewPos(newPosition);
        for (size_t k = 0; k < NKernels; ++k) {
            PacketVec3 origin = packetCurPos + kernels[k].mu * distances[k];
            PacketVec3 po = origin - packetNewPos;
            Packet t = std::experimental::sqrt(dot(po, po));
            Mask mask = std::experimental::isfinite(distances[k]) && t > 0;
            set(out.kernels[k].mu, po / t, mask);
            out.kernels[k].alpha = kernels[k].alpha;
            out.kernels[k].kappa = kernels[k].kappa;
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

    inline static void set(PacketVec3 & pv, size_t i, const Vector3 & v) {
        pv[0][i] = v[0];
        pv[1][i] = v[1];
        pv[2][i] = v[2];
    }

    inline static void set(Packet & pk1, const Packet & pk2, const Mask & mask) {
        for (size_t i = 0; i < NScalars; ++i) {
            if (mask[i]) {
                pk1[i] = pk2[i];
            }
        }
    }

    inline static void set(PacketVec3 & pv1, const PacketVec3 & pv2, const Mask & mask) {
        for (size_t i = 0; i < NScalars; ++i) {
            if (mask[i]) {
                pv1[0][i] = pv2[0][i];
                pv1[1][i] = pv2[1][i];
                pv1[2][i] = pv2[2][i];
            }
        }
    }

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
        return std::experimental::clamp(kappa, Packet(1e-2f), Packet(1e+4f));
    }

    inline void warpTo(const Vector3 & newPosition) {
        PacketVec3 packetCurPos(currentPosition);
        PacketVec3 packetNewPos(newPosition);
        for (size_t k = 0; k < NKernels; ++k) {
            PacketVec3 origin = packetCurPos + kernels[k].mu * distances[k];
            PacketVec3 po = origin - packetNewPos;
            Packet t = std::experimental::sqrt(dot(po, po));
            Mask mask = std::experimental::isfinite(distances[k]) && t > 0;
            // update distance, update mu
            set(distances[k], t, mask);
            set(kernels[k].mu, po / t, mask);
        }
        currentPosition = newPosition;
    }

    void updateKernels(SampleIterator begin, SampleIterator end) {
        // compute previous sufficient statistics
        Packet lastGammaWeightSums[NKernels];
        PacketVec3 lastGammaWeightSampleSums[NKernels];
        for (size_t k = 0; k < NKernels; ++k) {
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
            size_t i = 0;
            for (auto iter = begin; iter != end; ++iter) {
                Scalar pdf = 0.f;
                for (size_t k = 0; k < NKernels; ++k) {
                    partialPdfs[k] = kernels[k].alpha * kernels[k].pdf(iter->direction);
                    pdf += std::experimental::reduce(partialPdfs[k], std::plus<>());
                }

                // TODO sometimes encounter samples with zero pdf (value is too small)
                pdf = std::max(pdf, std::numeric_limits<Scalar>::min());

                Scalar weight = iter->radiance / iter->pdf;
                for (size_t k = 0; k < NKernels; ++k) {
                    Packet gammaIK = partialPdfs[k] / pdf;
                    batchGammaWeightSums[k] += gammaIK * weight;
                    batchGammaWeightSampleSums[k] += gammaIK * weight * PacketVec3(iter->direction);
                }

                logLikelihood += weight * std::log(pdf);
                i += 1;
            }

            // update parameters
            for (size_t k = 0; k < NKernels; ++k) {
                Packet gammaWeightSum = mix(movingWeight, lastGammaWeightSums[k], batchGammaWeightSums[k]);
                PacketVec3 gammaWeightSampleSum = mix(movingWeight, lastGammaWeightSampleSums[k], batchGammaWeightSampleSums[k]);
                Packet rLength = std::experimental::sqrt(dot(gammaWeightSampleSum, gammaWeightSampleSum));

                Mask mask = gammaWeightSum > 0 && rLength > 0;

                set(kernels[k].alpha, gammaWeightSum / sampleWeightSum, mask);
                set(kernels[k].mu, gammaWeightSampleSum / rLength, mask);
                set(meanCosine[k], std::experimental::min(rLength / gammaWeightSum, Packet(0.9999f)), mask);
                set(kernels[k].kappa, meanCosineToKappa(meanCosine[k]), mask);
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
            for (size_t k = 0; k < NKernels; ++k) {
                componentPdfs[k] = kernels[k].pdf(iter->direction);
                pdf += std::experimental::reduce(kernels[k].alpha * componentPdfs[k], std::plus<>());
            }

            for (size_t k = 0; k < NKernels; ++k) {
                Packet gammaIK = kernels[k].alpha * componentPdfs[k] / pdf;
                Packet distanceWeight = weight * gammaIK * componentPdfs[k];
                batchWeightedDistanceSums[k] += distanceWeight / iter->distance;
                batchDistanceWeightSums[k] += distanceWeight;
            }
        }

        Scalar movingWeight = 1.f / batchIndex;
        for (size_t k = 0; k < NKernels; ++k) {
            Mask mask = batchDistanceWeightSums[k] > 0 && batchWeightedDistanceSums[k] > 0;
            Packet weightedDistanceSum = distanceWeightSums[k] / distances[k];
            Packet newWeightSum = mix(movingWeight, distanceWeightSums[k], batchDistanceWeightSums[k]);
            Packet newWeightedDistanceSum = mix(movingWeight, weightedDistanceSum, batchWeightedDistanceSums[k]);
            set(distances[k], newWeightSum / newWeightedDistanceSum, mask);
            set(distanceWeightSums[k], newWeightSum, mask);
        }
    }
};

}

MTS_NAMESPACE_END
