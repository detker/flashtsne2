#ifndef NODE
#define NODE

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>

class Node {
public:
    using NodePtr = std::unique_ptr<Node>;

    // 1. Base constructor for a leaf (no children)
    explicit Node(std::string t) : tag(std::move(t)) {}

    // 2. Variadic constructor: allows Node("tag", move(child1), move(child2)...)
    template<typename... Args>
    Node(std::string t, Args&&... args) : tag(std::move(t)) {
        // This "folds" the arguments into the vector
        (children.push_back(std::forward<Args>(args)), ...);
    }

    void dump(size_t depth = 0) const {
        for(size_t i = 0; i < depth; i++) std::cout << "  ";
        std::cout << tag << "\n";

        for(const auto& child : children) {
            if (child) child->dump(depth + 1);
        }
    }

    std::string tag;
    std::vector<NodePtr> children;
};

// Helper function to create pointers easily
template<typename... Args>
std::unique_ptr<Node> make_node(Args&&... args) {
    return std::make_unique<Node>(std::forward<Args>(args)...);
}

#endif