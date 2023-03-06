#pragma once
#include "vmm.h"
#include "samplingfraction.h"
#include "guidedbsdf.h"
#include <queue>

MTS_NAMESPACE_BEGIN

class BSTree {
public:

    inline BSTree() {
        root = new Leaf(new VMM(), new AdamBSDFSamplingFraction());
    }

    virtual ~BSTree() {
        delete root;
    }

    inline GuidedBSDF guidedBSDF(const Vector3 & position, const BSDF * bsdf) {
        // down to the leaf node
        Node * currentNode = root;
        while (!currentNode->isLeaf) {
            auto * branch = (Branch *) currentNode;
            int index = position[branch->splitAxis] >= branch->splitPos;
            currentNode = branch->children[index];
        }

        // wrap the distribution
        auto * leaf = (Leaf *) currentNode;
        return {
            position,
            bsdf,
            leaf->directionModel,
            leaf->bsdfFractionModel
        };
    }

    inline void update(std::vector<PGSamplingRecord> & samples) {
#pragma omp parallel default(none), shared(samples)
#pragma omp single nowait
        root = root->update(samples, 0, (int) samples.size());
    }

    void reportStatistics() const {
        // count the number of nodes
        int numNodes = 0;
        int numLeaves = 0;
        std::queue<Node *> nodes;
        nodes.push(root);
        while (!nodes.empty()) {
            Node * currentNode = nodes.front();
            nodes.pop();
            numNodes += 1;
            numLeaves += currentNode->isLeaf;
            if (!currentNode->isLeaf) {
                nodes.push(((Branch *) currentNode)->children[0]);
                nodes.push(((Branch *) currentNode)->children[1]);
            }
        }

        Float MB = 1024.f * 1024;
        Float memoryForInner = (numNodes - numLeaves) * sizeof(Branch) / MB;
        Float memoryForLeaves = numLeaves * sizeof(Leaf) / MB;
        Float memoryForVMMs = numLeaves * sizeof(VMM) / MB;
        Float memoryForFractionModel = numLeaves * sizeof(AdamBSDFSamplingFraction) / MB;
        Float memoryTotal = memoryForInner + memoryForLeaves + memoryForVMMs + memoryForFractionModel;

        std::cout << "BSTree Statistics {" << std::endl
                  << "  Number of all nodes: " << numNodes << std::endl
                  << "  Number of leaf nodes: " << numLeaves << std::endl
                  << "  Memory for inner nodes: " << memoryForInner << " MB" << std::endl
                  << "  Memory for leaf nodes: " << memoryForLeaves << " MB" << std::endl
                  << "  Memory for direction model: " << memoryForVMMs << " MB" << std::endl
                  << "  Memory for fraction model: " << memoryForFractionModel << " MB" << std::endl
                  << "  Estimated total memory: " << memoryTotal << " MB" << std::endl
                  << "}";
    }

private:

    struct Node {

        bool isLeaf;

        inline explicit Node(bool isLeaf) {
            this->isLeaf = isLeaf;
        }

        [[nodiscard]]
        virtual Node * update(std::vector<PGSamplingRecord> & samples, int start, int end) = 0;

        virtual ~Node() = default;

    };

    struct Branch : public Node {
        Node * children[2]{};
        Float splitPos;
        int splitAxis;

        inline Branch(Node * left, Node * right, Float splitPos, int splitAxis) : Node(false) {
            this->children[0] = left;
            this->children[1] = right;
            this->splitPos = splitPos;
            this->splitAxis = splitAxis;
        }

        ~Branch() override {
            delete children[0];
            delete children[1];
        }

        [[nodiscard]]
        Node * update(std::vector<PGSamplingRecord> & samples, int start, int end) override {
            if (start < end) {
                // divide the samples
                int middle = start;
                while (samples[middle].position[splitAxis] < splitPos) {
                    if (++middle >= end) {
                        break;
                    }
                }
                for (int i = middle; i < end; ++i) {
                    if (samples[i].position[splitAxis] < splitPos) {
                        std::swap(samples[middle], samples[i]);
                        middle += 1;
                    }
                }

#pragma omp task mergeable default(none), shared(samples), firstprivate(start, middle)
                children[0] = children[0]->update(samples, start, middle);
                children[1] = children[1]->update(samples, middle, end);
            }
            return this;
        }
    };

    struct Leaf : public Node {

        constexpr static int MAX_NUM_SAMPLES = 32768;
//        constexpr static int MAX_NUM_SAMPLES = 2147483647;

        Vector3 positionMean;
        Vector3 positionVariance;
        int numSamples{};
        VMM * directionModel;
        AdamBSDFSamplingFraction * bsdfFractionModel;

        inline explicit Leaf(VMM * directionModel, AdamBSDFSamplingFraction * bsdfFractionModel) : Node(true) {
            positionMean = Vector3(0.f);
            positionVariance = Vector3(0.f);
            numSamples = 0;
            this->directionModel = directionModel;
            this->bsdfFractionModel = bsdfFractionModel;
        }

        ~Leaf() override {
            delete directionModel;
            delete bsdfFractionModel;
        }

        [[nodiscard]]
        Node * update(std::vector<PGSamplingRecord> & samples, int start, int end) override {
            if (start >= end) {
                return this;
            }

            // update statistics
            if (numSamples == 0 && start < end) {
                positionMean = samples[start].position;
                positionVariance = Vector3(0.f);
                numSamples += 1;
            }
            for (int i = start + 1; i < end; ++i) {
                Vector3 diff = samples[i].position - positionMean;
                positionMean += diff / (numSamples + 1.f);
                positionVariance += math::mul(diff, diff) / (numSamples + 1.f) - positionVariance / numSamples;
                numSamples += 1;
            }

            // check if a split is required
            if (numSamples >= MAX_NUM_SAMPLES) {
                int splitAxis = math::argmax(positionVariance);
                Float splitPos = positionMean[splitAxis];

                // TODO better handling of statistics on spatial splitting
                positionMean = Vector3(0.f);
                positionVariance = Vector3(0.f);
                numSamples = 0;
                directionModel->retarget();
                bsdfFractionModel->retarget();

                // create a sibling node
                auto * leaf = new Leaf(new VMM(*directionModel), new AdamBSDFSamplingFraction(*bsdfFractionModel));
                auto * branch = new Branch(this, leaf, splitPos, splitAxis);
                return branch->update(samples, start, end);
            }

            // update the model
            bsdfFractionModel->update(samples, start, end);
            directionModel->update(samples, start, end, positionMean);
            return this;
        }
    };

    Node * root;

};

MTS_NAMESPACE_END
