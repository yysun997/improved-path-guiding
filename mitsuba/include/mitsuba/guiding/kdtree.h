#pragma once
#include <mutex>
#include <vector>
#include "guidedbsdf.h"
#include <omp.h>

MTS_NAMESPACE_BEGIN

namespace PathGuiding::kdtree {

using SampleIterator = std::vector<SampleData>::iterator;
using ParallaxAwareVMM = vmm::ParallaxAwareVMM;

class Region {
public:

    Vector3 posMean;
    Vector3 posVar;
    ParallaxAwareVMM * model{};
    size_t numSamples{};

    inline explicit Region(ParallaxAwareVMM * model) {
        this->model = model;
    }

    virtual ~Region() {
        delete model;
    }

    inline static Vector3 square(const Vector3 & v) {
        return {v.x * v.x, v.y * v.y, v.z * v.z};
    }

    inline void updateStats(SampleIterator begin, SampleIterator end) {
        auto iter = begin;
        if (numSamples == 0) {
            posMean = iter->position;
            posVar = Vector3(0.f);
            numSamples += 1;
            ++iter;
        }
        while (iter != end) {
            Vector3 diff = iter->position - posMean;
            posMean += diff / (numSamples + 1.f);
            posVar += square(diff) / (numSamples + 1.f) - posVar / numSamples;
            numSamples += 1;
            ++iter;
        }
    }

    inline void clearStats() {
        numSamples = 0;
        posMean = Vector3(0.f);
        posVar = Vector3(0.f);
    }
};

class AdaptiveKDTree {
private:
    
    struct Node {
        union {
            struct { Node * children[2]{}; };
            struct { Node * parent; Region * region; };
        };
        Float splitPos;
        int splitAxis;

        inline Node(Node * left, Node * right, int splitAxis, Float splitPos) {
            this->children[0] = left;
            this->children[1] = right;
            this->splitAxis = splitAxis;
            this->splitPos = splitPos;
        }

        inline Node(Region * region, Node * parent) {
            this->region = region;
            this->parent = parent;
            this->splitAxis = -1;
            this->splitPos = 0;
        }

        virtual ~Node() {
            if (splitAxis < 0) {
                delete region;
            } else {
                delete children[0];
                delete children[1];
            }
        }
    };

    Node * root{};

public:

    inline AdaptiveKDTree() {
        auto model = new ParallaxAwareVMM();
        auto region = new Region(model);
        root = new Node(region, nullptr);
    }

    virtual ~AdaptiveKDTree() {
        delete root;
    }

    inline void getGuidedBSDF(GuidedBSDF & guidedBSDF, const BSDF * bsdf, const Vector3 & position) const {
        Node * currentNode = root;
        while (currentNode->splitAxis >= 0) {
            int childIdx = (position[currentNode->splitAxis] < currentNode->splitPos ? 0 : 1);
            currentNode = currentNode->children[childIdx];
        }

        auto region = currentNode->region;
        region->model->getWarped(guidedBSDF.model, position);
        guidedBSDF.bsdf = bsdf;
    }

    inline void update(std::vector<SampleData> & samples) {
        size_t nCores = Scheduler::getInstance()->getCoreCount();
        Thread::initializeOpenMP(nCores);

#pragma omp parallel default(none), shared(samples)
#pragma omp single nowait
        updateNode(root, samples.begin(), samples.end());
    }

private:

    inline static int argmax(const Vector3 & v) {
        return v[0] >= v[1] ? (v[0] >= v[2] ? 0 : 2) : (v[1] >= v[2] ? 1 : 2);
    }

    static void updateNode(Node * node, SampleIterator begin, SampleIterator end) {
        if (begin >= end) {
            return;
        }

        const size_t maxRegionNumSamples = 32768;

        if (node->splitAxis < 0) {
            auto region = node->region;
            region->updateStats(begin, end);
            if (region->numSamples < maxRegionNumSamples) {
                region->model->update(region->posMean, begin, end);
                return;
            } else {
                node->splitAxis = argmax(region->posVar);
                node->splitPos = region->posMean[node->splitAxis];
                region->clearStats();

                auto leftChild = new Node(region, node);
                auto rightRegion = new Region(region->model->split());
                auto rightChild = new Node(rightRegion, node);

                node->children[0] = leftChild;
                node->children[1] = rightChild;
            }
        }

        if (node->splitAxis >= 0) {
            auto middle = std::partition(begin, end, [&](const SampleData & sample) -> bool {
                return sample.position[node->splitAxis] < node->splitPos;
            });

#pragma omp task mergeable default(none), firstprivate(node, begin, middle)
            updateNode(node->children[0], begin, middle);
#pragma omp task mergeable default(none), firstprivate(node, middle, end)
            updateNode(node->children[1], middle, end);
        }
    }
};

}

MTS_NAMESPACE_END
