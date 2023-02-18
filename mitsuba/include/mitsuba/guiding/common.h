#pragma once
#include "mitsuba/mitsuba.h"
#include <vector>

MTS_NAMESPACE_BEGIN

struct PGSamplingRecord {
    Vector3 position;
    Vector3 direction;
    Float radiance{};
    Float pdf{};

    Float product{};
    Float bsdfSamplingFraction{};
    Float pdfBSDF{};

    inline explicit PGSamplingRecord(const Point3 & position)
    {
        this->position = Vector3(position);
    }
};

namespace math {

template <typename T>
inline T mix(T t, T a, T b) {
    return (1.f - t) * a + t * b;
}

template <typename T>
inline TVector3<T> mix(T t, const TVector3<T> & v1, const TVector3<T> & v2) {
    return (1.f - t) * v1 + t * v2;
}

template <typename T>
inline TVector3<T> mul(const TVector3<T> & v1, const TVector3<T> & v2) {
    return {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
}

template <typename T>
inline int argmax(const TVector3<T> & v) {
    return v[0] >= v[1] ? (v[0] >= v[2] ? 0 : 2) : (v[1] >= v[2] ? 1 : 2);
}

}

MTS_NAMESPACE_END
