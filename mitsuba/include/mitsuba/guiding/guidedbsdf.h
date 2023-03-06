#pragma once
#include "mitsuba/render/bsdf.h"
#include "vmm.h"

MTS_NAMESPACE_BEGIN

class GuidedBSDF {
public:

    inline GuidedBSDF(const Vector3 & position, const BSDF * bsdf, VMM * model, AdamBSDFSamplingFraction * bsdfFractionModel) {
        this->position = position;
        this->bsdf = bsdf;
        this->directionModel = model;
        this->bsdfFractionModel = bsdfFractionModel;
    }

    Spectrum sample(BSDFSamplingRecord & bRec, PGSamplingRecord & pgRec, const Point2 & sample) {
        pgRec.bsdfSamplingFraction = bsdfFractionModel->bsdfSamplingFraction();
        if (sample.x < pgRec.bsdfSamplingFraction) {
            // sample BSDF
            Spectrum result = bsdf->sample(bRec, pgRec.pdfBSDF, {sample.x / pgRec.bsdfSamplingFraction, sample.y});
            if (result.isZero()) {
                return result;
            }
            result *= pgRec.pdfBSDF;

//            assert(!std::isnan(bRec.wo[0]) && !std::isnan(bRec.wo[1]) && !std::isnan(bRec.wo[2]));

            // one-sample MIS
            pgRec.direction = bRec.its.toWorld(bRec.wo);
//            assert(!std::isnan(pgRec.direction[0]) && !std::isnan(pgRec.direction[1]) && !std::isnan(pgRec.direction[2]));
            Float pdfModel = directionModel->pdf(pgRec.direction, position);
            pgRec.pdf = math::mix(pgRec.bsdfSamplingFraction, pdfModel, pgRec.pdfBSDF);
            return result / pgRec.pdf;
        }

        // sample model
        Float pdfModel;
        const Point2 reusedSample = {(sample.x - pgRec.bsdfSamplingFraction) / (1 - pgRec.bsdfSamplingFraction), sample.y};
        pgRec.direction = directionModel->sample(reusedSample, pdfModel, position);
        bRec.wo = bRec.its.toLocal(pgRec.direction);
        Spectrum result = bsdf->eval(bRec);
        if (result.isZero()) {
            return result;
        }

        // one-sample MIS
        pgRec.pdfBSDF = bsdf->pdf(bRec);
        pgRec.pdf = math::mix(pgRec.bsdfSamplingFraction, pdfModel, pgRec.pdfBSDF);
        return result / pgRec.pdf;
    }

    Float pdf(const BSDFSamplingRecord & bRec) {
        Vector3 wo = bRec.its.toWorld(bRec.wo);
        if (std::isnan(wo[0]) || std::isnan(wo[1]) || std::isnan(wo[2])) {
            return 0;
        }

        Float bsdfSamplingFraction = bsdfFractionModel->bsdfSamplingFraction();
        Float pdfModel = directionModel->pdf(wo, position);
        Float pdfBSDF = bsdf->pdf(bRec);
        return math::mix(bsdfSamplingFraction, pdfModel, pdfBSDF);
    }

private:

    Vector3 position;
    const BSDF * bsdf;
    VMM * directionModel;
    AdamBSDFSamplingFraction * bsdfFractionModel;

};

MTS_NAMESPACE_END
