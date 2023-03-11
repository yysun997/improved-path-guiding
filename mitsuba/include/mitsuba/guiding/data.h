#pragma once
#include "mitsuba/mitsuba.h"

MTS_NAMESPACE_BEGIN

namespace PathGuiding {

struct SampleData {
    Vector3 position;
    Vector3 direction;
    Float radiance{};
    Float pdf{};
    Float distance{};

    inline explicit SampleData(const Vector3 & position, const Vector3 & direction,
        Float radiance, Float pdf, Float distance) {
        this->position = Vector3(position);
        this->direction = direction;
        this->radiance = radiance;
        this->pdf = pdf;
        this->distance = distance;
    }

    [[nodiscard]]
    inline bool isValid() const {
        if (position.isNaN() || direction.isNaN() || direction.isZero()) {
            return false;
        }
        if (std::isnan(radiance) || radiance <= 0) {
            return false;
        }
        if (std::isnan(pdf) || pdf <= 0) {
            return false;
        }
        if (std::isnan(distance) || distance <= 0) {
            return false;
        }
        return true;
    }
};

}

MTS_NAMESPACE_END
