#include <cassert>
#include <cstdio>

#include <crystalnet/utility/enumerate.hpp>
#include <teavana/range.hpp>

void test_1()
{
    for (auto[idx, n] : enumerate(tea::range(100, 110))) {
        assert(idx + 100 == n);
        printf("%d %d\n", idx, n);
    }
}

int main()
{
    test_1();
    return 0;
}
