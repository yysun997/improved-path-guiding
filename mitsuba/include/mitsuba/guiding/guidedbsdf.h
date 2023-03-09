#pragma once
#include "mitsuba/render/bsdf.h"
#include "vmm.h"

MTS_NAMESPACE_BEGIN

namespace PathGuiding {

// TODO decide sampling fraction based on roughness?
class GuidedBSDF {
public:

    using VMFMixture = vmm::VMFMixture;

    VMFMixture model;
    const BSDF * bsdf{};

    Spectrum sample(BSDFSamplingRecord & bRec, Float & pdf, const Point2 & rn) const {
        Float bsdfSamplingFraction = 0.5;
        if (rn.x < bsdfSamplingFraction) {
            // sample the BSDF
            Float pdfBSDF;
            Spectrum result = bsdf->sample(bRec, pdfBSDF, {rn.x / bsdfSamplingFraction, rn.y});
            if (result.isZero()) {
                return result;
            }
            result *= pdfBSDF;

            // one-sample MIS
            Vector3 direction = bRec.its.toWorld(bRec.wo);
            Float pdfModel = model.pdf(direction);
            pdf = math::lerp(bsdfSamplingFraction, pdfModel, pdfBSDF);
            return result / pdf;
        }

        // sample the model
        const Vector2 reusedSample = {(rn.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction), rn.y};
        Vector3 direction = model.sample(reusedSample);
        Float pdfModel = model.pdf(direction);
        bRec.wo = bRec.its.toLocal(direction);
        Spectrum result = bsdf->eval(bRec);
        if (result.isZero()) {
            return result;
        }

        // one-sample MIS
        Float pdfBSDF = bsdf->pdf(bRec);
        pdf = math::lerp(bsdfSamplingFraction, pdfModel, pdfBSDF);
        return result / pdf;
    }

    [[nodiscard]]
    Float pdf(const BSDFSamplingRecord & bRec) const {
        Float bsdfSamplingFraction = 0.5;
        Vector3 wo = bRec.its.toWorld(bRec.wo);
        Float pdfModel = model.pdf(wo);
        Float pdfBSDF = bsdf->pdf(bRec);
        return math::lerp(bsdfSamplingFraction, pdfModel, pdfBSDF);
    }
};

}

MTS_NAMESPACE_END
