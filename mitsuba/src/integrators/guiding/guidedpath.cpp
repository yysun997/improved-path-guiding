#include "mitsuba/render/scene.h"
#include "mitsuba/core/statistics.h"
#include "mitsuba/render/renderproc.h"
#include "mitsuba/guiding/guiding.h"
#include <condition_variable>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracer : public MonteCarloIntegrator {
public:

    inline explicit GuidedPathTracer(const Properties & props)
        : MonteCarloIntegrator(props)
    {
        m_tree = std::make_shared<BSTree>();
        m_numFlushedSamples = 0;
        m_training = true;
        m_firstIteration = true;

        m_sppPerIteration = props.getInteger("sppPerIteration", 4);
        m_minTrainingSPPFraction = props.getFloat("minTrainingSPPFraction", 0.5);

        // in megabytes
        m_maxSampleBufferSize = props.getInteger("maxSampleBufferSize", 256);
        m_maxSampleBufferSize = m_maxSampleBufferSize * (1024 * 1024 / sizeof(PGSamplingRecord));
    }

    ~GuidedPathTracer() override = default;

    bool render(Scene * scene, RenderQueue * queue, const RenderJob * job,
        int sceneResID, int sensorResID, int samplerResID) override
    {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = (Sensor *) sched->getResource(sensorResID);
        ref<Film> film = sensor->getFilm();

        size_t nCores = sched->getCoreCount();
        auto * sampler = (const Sampler *) sched->getResource(samplerResID, 0);
        size_t sampleCount = sampler->getSampleCount();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

        bool success = true;

        // TODO better criteria for stopping training?
        const int minTrainingSPP = (int) (m_minTrainingSPPFraction * sampleCount);
        const int trainingWindowSize = 5;
        const int minNumSamplesToTrain = 128;
        const Float minGrowthRate = 2e-3;

        /* This is a sampling-based integrator - parallelize */
        int iteration = 0;
        int sppRendered = 0;
        int numValidSamples[trainingWindowSize];
        int integratorResID = sched->registerResource(this);
        while (sppRendered < sampleCount && m_training) {
            if (sppRendered + m_sppPerIteration > sampleCount) {
                m_sppPerIteration = (int) sampleCount - sppRendered;
            }

            m_numFlushedSamples = 0;

            ref<ParallelProcess> proc = new BlockedRenderProcess(job, queue, scene->getBlockSize());
            proc->bindResource("integrator", integratorResID);
            proc->bindResource("scene", sceneResID);
            proc->bindResource("sensor", sensorResID);
            proc->bindResource("sampler", samplerResID);
            scene->bindUsedResources(proc);
            bindUsedResources(proc);
            sched->schedule(proc);

            m_process = proc;
            sched->wait(proc);
            m_process = nullptr;

            iteration += 1;
            sppRendered += m_sppPerIteration;
            if (proc->getReturnStatus() != ParallelProcess::ESuccess) {
                success = false;
                break;
            }

            int numSamplesInIteration = m_numFlushedSamples + m_samples.size();
            Log(EInfo, "got %d samples in iteration %d", numSamplesInIteration, iteration);

            numValidSamples[(iteration - 1) % trainingWindowSize] = numSamplesInIteration;
            if (iteration >= trainingWindowSize) {
                int minValue = std::numeric_limits<int>::max();
                int maxValue = std::numeric_limits<int>::min();
                for (int num: numValidSamples) {
                    minValue = std::min(minValue, num);
                    maxValue = std::max(maxValue, num);
                }

                Float avgFiveGrowthRate = (Float) (maxValue - minValue) / minValue;
                if ((sppRendered >= minTrainingSPP && avgFiveGrowthRate < minGrowthRate) || sppRendered == sampleCount) {
                    Log(EInfo, "stop training after iteration %d", iteration);
                    m_training = false;
                }
            }

            if (m_samples.size() >= minNumSamplesToTrain) {
                if (m_training) {
                    m_tree->update(m_samples);
                    m_firstIteration = false;
                }
                m_samples.clear();
            } else {
                Log(EInfo, "skip training due to insufficient samples");
            }
        }

        std::vector<PGSamplingRecord>().swap(m_samples);

        if (success && sppRendered < sampleCount) {
            m_sppPerIteration = sampleCount - sppRendered;
            Log(EInfo, "rendering for the rest %d spp", m_sppPerIteration);

            ref<ParallelProcess> proc = new BlockedRenderProcess(job, queue, scene->getBlockSize());
            proc->bindResource("integrator", integratorResID);
            proc->bindResource("scene", sceneResID);
            proc->bindResource("sensor", sensorResID);
            proc->bindResource("sampler", samplerResID);
            scene->bindUsedResources(proc);
            bindUsedResources(proc);
            sched->schedule(proc);

            m_process = proc;
            sched->wait(proc);
            m_process = nullptr;

            success = proc->getReturnStatus() == ParallelProcess::ESuccess;
        }
        sched->unregisterResource(integratorResID);
        m_tree->reportStatistics();
        Log(EInfo, "average distance = %.3f\n", totalDistance / distanceCount);

        return success;
    }

    // TODO use a better combine strategy
    void renderBlock(const Scene * scene, const Sensor * sensor, Sampler * sampler, ImageBlock * block,
        const bool & stop, const std::vector< TPoint2<uint8_t> > & points) const override
    {
        Float diffScaleFactor = 1.0f / std::sqrt((Float) sampler->getSampleCount());

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        int queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) {
            /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;
        }

        for (auto & point : points) {
            Point2i offset = Point2i(point) + Vector2i(block->getOffset());
            if (stop) {
                break;
            }

            sampler->generate(offset);

            for (size_t j = 0; j < m_sppPerIteration; j++) {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample) {
                    apertureSample = rRec.nextSample2D();
                }
                if (needsTimeSample) {
                    timeSample = rRec.nextSample1D();
                }

                Spectrum spec = sensor->sampleRayDifferential(sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                spec *= Li(sensorRay, rRec);
                block->put(samplePos, spec, rRec.alpha);
                sampler->advance();
            }
        }
    }

    Spectrum Li(const RayDifferential & r, RadianceQueryRecord & rRec) const override {
        /* Some aliases and local variables */
        const Scene * scene = rRec.scene;
        Intersection & its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        const auto miWeight = [](Float pdfA, Float pdfB) -> Float {
            return pdfA / (pdfA + pdfB);
        };

        const auto computeDistanceFactor = [](const BSDFSamplingRecord & bRec) -> Float {
            Float cosThetaI = Frame::cosTheta(bRec.wi);
            Float cosThetaO = Frame::cosTheta(bRec.wo);
            if (cosThetaI * cosThetaO <= 0) {
                Float sinThetaI = math::safe_sqrt(1 - cosThetaI * cosThetaI);
                if (sinThetaI > 0) {
                    Float sinThetaO = math::safe_sqrt(1 - cosThetaO * cosThetaO);
                    return sinThetaO / sinThetaI;
                }
            }
            return 1;
        };

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        // TODO check if this is correct
        if (!its.isValid()) {
            if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && !m_hideEmitters) {
                /* If no intersection could be found, potentially return
                radiance from a environment luminaire if it exists */
                return scene->evalEnvironment(ray);
            }
            return Spectrum(0.f);
        }

        if (its.isEmitter()) {
            if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && !m_hideEmitters) {
                return its.Le(-ray.d);
            }
            return Spectrum(0.f);
        }

        Spectrum throughput(1.0f);
        Float eta = 1.0f;
        std::vector<PGSamplingRecord> pgRecords;

        struct BounceInfo {
            Spectrum li{0.f};
            Spectrum scatterWeight{0.f};
            Float scatterPdf{0};
            Float distance{0};
            Float roughness{0};
            Float distanceFactor{1};
            bool usePathGuiding{false};
        };

        std::vector<BounceInfo> bounces;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
//            if (!its.isValid()) {
//                /* If no intersection could be found, potentially return
//                   radiance from a environment luminaire if it exists */
//                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered)) {
//                    Li += throughput * scene->evalEnvironment(ray);
//                }
//                break;
//            }

            pgRecords.emplace_back(its.p);
            PGSamplingRecord & pgRec = pgRecords.back();
            bounces.emplace_back();
            BounceInfo & bInfo = bounces.back();

            const BSDF * bsdf = its.getBSDF();
            auto guidedBSDF = m_tree->guidedBSDF(pgRec.position, bsdf);

            bInfo.roughness = bsdf->getRoughness(its);
            assert(bInfo.roughness >= 0 && bInfo.roughness <= 1);
            bInfo.usePathGuiding = !m_firstIteration && bInfo.roughness >= 0.01;

            /* Possibly include emitted radiance if requested */
//            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
//                && (!m_hideEmitters || scattered)) {
//                Spectrum radiance = its.Le(-ray.d);
//                Li += throughput * radiance;
//                bInfo.li += radiance;
//            }

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                Spectrum radiance = its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
                Li += throughput * radiance;
                bInfo.li += radiance;
            }

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    auto * emitter = (const Emitter *) dRec.object;

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                                              || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = 0;
                        if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
                            if (bInfo.usePathGuiding) {
                                assert(!std::isnan(bRec.wo[0]) && !std::isnan(bRec.wo[1]) && !std::isnan(bRec.wo[2]));
                                bsdfPdf = guidedBSDF.pdf(bRec);
                            } else {
                                bsdfPdf = bsdf->pdf(bRec);
                            }
                        }
                        assert(!std::isnan(bsdfPdf));

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;

                        // TODO handle NEE?
                        bInfo.li += weight * bsdfVal * value;
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);

            Float bsdfPdf;
            Spectrum bsdfWeight;
            if (bInfo.usePathGuiding) {
                bsdfWeight = guidedBSDF.sample(bRec, pgRec, rRec.nextSample2D());
                bsdfPdf = pgRec.pdf;
            } else {
                bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
                pgRec.direction = bRec.its.toWorld(bRec.wo);
                pgRec.pdf = bsdfPdf;
            }
            bInfo.scatterPdf = bsdfPdf;

            if (bsdfWeight.isZero()) {
                break;
            }

            assert(!std::isnan(bsdfPdf));
            assert(!bsdfWeight.isNaN());

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0) {
                break;
            }

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter * env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered) {
                        break;
                    }

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray)) {
                        break;
                    }
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;
            bInfo.scatterWeight = bsdfWeight;
            bInfo.distanceFactor = computeDistanceFactor(bRec);

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter) {
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) {
                    /* Compute the prob. of generating that direction using the
                       implemented direct illumination sampling technique */
                    const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float weight = miWeight(bsdfPdf, lumPdf);
                    Li += throughput * value * weight;

                    bInfo.li += weight * bsdfWeight * value;

//                    if (m_training && bInfo.usePathGuiding) {
                    if (m_training) {
                        bInfo.distance = its.isValid() ? its.t : scene->getAABB().getBSphere().radius * 1024;
                        assert(bInfo.distance > 0);
                        if (bInfo.roughness >= 0.01) {
                            pgRec.radiance = value.average();
                            pgRec.product = (bsdfWeight * value * bsdfPdf).average();
                            pgRec.distance = bInfo.distance;
                            if (pgRec.radiance > 0 && pgRec.product > 0) {
                                addPGSamplingRecord(pgRec);
                            }
                        }
                    }
                }
                // stop tracing after hitting an emitter
                break;
            }

            bInfo.distance = its.t;
            assert(bInfo.distance > 0);

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
                break;
            }
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q) {
                    break;
                }
                throughput /= q;
                bInfo.scatterWeight /= q;
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        assert(bounces.size() == pgRecords.size());

        if (m_training) {
            Float distance = bounces.back().distance;
            if (!bounces.back().scatterWeight.isZero() && distance <= 0) {
                auto & bInfo = bounces.back();
                std::cout << "li = " << bInfo.li.toString() << ", "
                          << "scatterWeight = " << bInfo.scatterWeight.toString() << ", "
                          << "scatterPdf = " << bInfo.scatterPdf << ", "
                          << "distance = " << bInfo.distance << ", "
                          << "roughness = " << bInfo.roughness << ", "
                          << "distanceFactor = " << bInfo.distanceFactor
                          << std::endl;
                assert(false);
            }
            for (int i = (int) bounces.size() - 2; i >= 0; --i) {
                // compute the incident radiance
                pgRecords[i].radiance = bounces[i + 1].li.average();
                bounces[i].li += bounces[i + 1].li * bounces[i].scatterWeight;
                pgRecords[i].product = (bounces[i + 1].li * bounces[i].scatterWeight * bounces[i].scatterPdf).average();

                // compute the perceived distance
                if (bounces[i + 1].roughness >= 0.3) {
                    distance = bounces[i].distance;
                } else {
                    distance *= bounces[i + 1].distanceFactor;
                    distance += bounces[i].distance;
                }
                pgRecords[i].distance = distance;

                std::unique_lock<std::mutex> lock(distanceMutex);
                totalDistance += distance;
                distanceCount += 1;
                lock.unlock();

                assert(!std::isnan(distance) && std::isfinite(distance) && distance > 0);

//                if (bounces[i].usePathGuiding && pgRecords[i].radiance > 0 && pgRecords[i].product > 0) {
//                    addPGSamplingRecord(pgRecords[i]);
//                }
                if (bounces[i].roughness >= 0.01 && pgRecords[i].radiance > 0 && pgRecords[i].product > 0) {
                    addPGSamplingRecord(pgRecords[i]);
                }
            }
        }

        return Li;
    }

    std::string toString() const override {
        std::ostringstream oss;
        oss << "GuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "  sppPerIteration = " << m_sppPerIteration << endl
            << "  maxSampleBufferSize = " << m_maxSampleBufferSize << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:

    mutable std::mutex m_sampleMutex;
    mutable std::condition_variable m_sampleCV;
    mutable std::vector<PGSamplingRecord> m_samples;
    mutable std::shared_ptr<BSTree> m_tree;
    mutable int m_numFlushedSamples;
    mutable bool m_training;
    mutable bool m_firstIteration;

    mutable std::mutex distanceMutex;
    mutable double totalDistance{};
    mutable uint64_t distanceCount{};

    Float m_minTrainingSPPFraction;
    int m_sppPerIteration;
    int m_maxSampleBufferSize;

    inline void addPGSamplingRecord(const PGSamplingRecord & pgRec) const {
        if (pgRec.radiance <= 0 || pgRec.product <= 0 || pgRec.pdf < 0 || pgRec.pdfBSDF < 0) {
            std::cout << "pgRec.position = " << pgRec.position.toString() << std::endl
                      << "pgRec.direction = " << pgRec.direction.toString() << std::endl
                      << "pgRec.bsdfSamplingFraction = " << pgRec.bsdfSamplingFraction << std::endl
                      << "pgRec.radiance = " << pgRec.radiance << std::endl
                      << "pgRec.product = " << pgRec.product << std::endl
                      << "pgRec.pdf = " << pgRec.pdf << std::endl
                      << "pgRec.pdfBSDF = " << pgRec.pdfBSDF << std::endl;
            assert(false);
        }

        std::unique_lock<std::mutex> lock(m_sampleMutex);
        m_sampleCV.wait(lock, [this] { return m_samples.size() < m_maxSampleBufferSize; });
        m_samples.push_back(pgRec);

        if (m_samples.size() >= m_maxSampleBufferSize) {
            lock.unlock();
            m_tree->update(m_samples);
            m_numFlushedSamples += m_samples.size();
            m_samples.clear();
            m_sampleCV.notify_all();
        }
    }
};

MTS_IMPLEMENT_CLASS(GuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracer, "Guided path tracer")
MTS_NAMESPACE_END
