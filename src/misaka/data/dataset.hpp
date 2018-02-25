#pragma once
#include <cassert>
#include <memory>
#include <utility>

#include <misaka/core/tensor.hpp>

struct dataset_t {
    using item_t = std::pair<tensor_ref_t, tensor_ref_t>;

    virtual item_t next() = 0;

    // virtual item_t next_barch() {} // TODO: next_batch

    virtual bool has_next() const = 0;
    virtual ~dataset_t() {}
};

struct range_t {
    using item_t = dataset_t::item_t;
    dataset_t &ds;

    explicit range_t(dataset_t &ds) : ds(ds) {}

    struct iter_t {
        dataset_t *ds;
        std::unique_ptr<item_t> next;
        iter_t(dataset_t *ds) : ds(ds) { this->operator++(); }
        bool operator!=(const iter_t &it) const { return ds != it.ds; }
        void operator++()
        {
            if (ds && ds->has_next()) {
                auto item = new item_t(ds->next());
                next.reset(item);
            } else {
                ds = nullptr;
            }
        }
        item_t operator*() { return item_t(*next); }
    };

    iter_t begin()
    {
        if (ds.has_next()) {
            return iter_t(&ds);
        }
        return end();
    }

    iter_t end() { return iter_t(nullptr); }
};

inline range_t range(dataset_t &ds) { return range_t(ds); }

struct simple_dataset_t : dataset_t {
    using owner_t = std::unique_ptr<tensor_t>;

    int idx;
    uint32_t n;

    owner_t images;
    owner_t labels;

    simple_dataset_t(tensor_t *images, tensor_t *labels)
        : images(owner_t(images)), labels(owner_t(labels)), idx(0),
          n(images->shape.len())
    {
    }

    bool has_next() const override { return idx < n; }

    item_t next() override
    {
        assert(has_next());
        auto i = idx++;
        return item_t((*images)[i], (*labels)[i]);
    }
};
