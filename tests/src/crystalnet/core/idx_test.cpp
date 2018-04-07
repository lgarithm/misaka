#include <cassert>

#include <crystalnet.h>

template <typename> struct idx_type;

template <> struct idx_type<uint8_t> {
    static constexpr uint8_t type = 0x08;
};

template <> struct idx_type<int8_t> {
    static constexpr uint8_t type = 0x09;
};

template <> struct idx_type<int16_t> {
    static constexpr uint8_t type = 0x0B;
};

template <> struct idx_type<int32_t> {
    static constexpr uint8_t type = 0x0C;
};

template <> struct idx_type<float> {
    static constexpr uint8_t type = 0x0D;
};

template <> struct idx_type<double> {
    static constexpr uint8_t type = 0x0E;
};

template <typename T> void test_dtyped()
{
    const auto dtype = idx_type<T>::type;
    const shape_t *shape = new_shape(3, 2, 3, 4);
    const auto dim = shape_dim(shape);
    {
        auto *t = new_tensor(shape, dtype);
        T *data = reinterpret_cast<T *>(tensor_data_ptr(tensor_ref(t)));
        for (int i = 0; i < dim; ++i) {
            data[i] = i;
        }
        save_tensor("test.idx", tensor_ref(t));
        del_tensor(t);
    }
    {
        auto t = _load_idx_file("test.idx");
        T *data = reinterpret_cast<T *>(tensor_data_ptr(tensor_ref(t)));
        assert(shape_dim(tensor_shape(tensor_ref(t))) == dim);
        for (int i = 0; i < dim; ++i) {
            assert(data[i] == i);
        }
        del_tensor(t);
    }
    del_shape(shape);
}

void test_1()
{
    test_dtyped<uint8_t>();
    test_dtyped<int8_t>();
    test_dtyped<int16_t>();
    test_dtyped<int32_t>();
    test_dtyped<float>();
    test_dtyped<double>();
}

int main()
{
    test_1();
    return 0;
}