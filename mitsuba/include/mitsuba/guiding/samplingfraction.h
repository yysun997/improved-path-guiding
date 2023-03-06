#pragma once
#include "mitsuba/mitsuba.h"
#include "common.h"

MTS_NAMESPACE_BEGIN

// for this implementation see Muller et al. 2019 "Path Guiding in Production"
// TODO bad on the scene "clocks", may be replaced by a heuristic function based on roughness?
class AdamBSDFSamplingFraction {
public:

    constexpr static Float EPSILON = 1e-8;
    constexpr static int BATCH_SIZE = 64;

    inline AdamBSDFSamplingFraction() {
        param = 0;
        firstMoment = 0;
        secondMoment = 0;
        beta1 = 0.99;
        beta2 = 0.999;
        learningRate = 0.01;
        regularization = 0.01;
        batchIndex = 0;
    }

    void update(const std::vector<PGSamplingRecord> & samples, int start, int end) {
        return;
//        int batchStart = start;
//        while (batchStart < end) {
//            int batchEnd = std::min(batchStart + BATCH_SIZE, end);
//
//            // compute the gradients
//            Float gradients = 0;
//            for (int i = batchStart; i < batchEnd; ++i) {
//                Float pdfModel = (samples[i].pdf - samples[i].bsdfSamplingFraction * samples[i].pdfBSDF) /
//                                 (1 - samples[i].bsdfSamplingFraction);
//                gradients += -samples[i].product * (samples[i].pdfBSDF - pdfModel) /
//                             (samples[i].pdf * samples[i].pdf) *
//                             samples[i].bsdfSamplingFraction * (1 - samples[i].bsdfSamplingFraction) +
//                             regularization * param;
//            }
//            gradients /= (Float) (batchEnd - batchStart);
//
//            // perform an Adam step
//            batchIndex += 1;
//            Float step = learningRate * std::sqrt(1 - std::pow(beta2, batchIndex)) / (1 - std::pow(beta1, batchIndex));
//            firstMoment = math::mix(beta1, gradients, firstMoment);
//            secondMoment = math::mix(beta2, gradients * gradients, secondMoment);
//            param -= step * firstMoment / (std::sqrt(secondMoment) + EPSILON);
//
//            batchStart = batchEnd;
//        }
    }

    inline Float bsdfSamplingFraction() const {
//        return 1.f / (1.f + std::exp(-param));
        return 0.5;
    }

    inline void retarget() {
        firstMoment *= 0.5;
        secondMoment *= 0.5 * 0.5;
        batchIndex = 0;
    }

private:

    Float param;
    Float firstMoment;
    Float secondMoment;
    Float beta1;
    Float beta2;
    Float learningRate;
    Float regularization;
    int batchIndex;

};

MTS_NAMESPACE_END
