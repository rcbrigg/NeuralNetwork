#pragma once
#include <stdint.h>
#include <cassert>
#include <initializer_list>
#include <array>

namespace nn
{
template<size_t N = 0>
class Shape
{
public:
	Shape()
	{
		size_ = 0;
		memset(dimensions_.data(), 0, sizeof(dimensions_) * N);
	}

	template<size_t M> Shape(const uint32_t(&dimensions)[M])
	{
		static_assert(M == N, "Wrong number of dimensions given to Shape<> ctor.");

		size_ = 1;
		for (size_t i = 0; i < N; ++i)
		{
			dimensions_[i] = dimensions[i];
			size_ *= dimensions[i];
		}

		assert(size_ != 0);
	}

	template<size_t M> Shape& operator = (const uint32_t(&dimensions)[M])
	{
		size_ = 1;
		for (size_t i = 0; i < N; ++i)
		{
			dimensions_[i] = dimensions[i];
			size_ *= dimensions[i];
		}
		return *this;
	}

	size_t size() const { return size_; }
	
	const auto& dimensions() const { return dimensions_; }

	size_t length(uint32_t dim = 0) const
	{
		assert(dim < N);
		return dimensions()[dim];
	}

	auto slice() const { return Shape<N - 1>(dimensions(), size_); }

private:
	friend Shape<N+1>;

	Shape(const std::array<uint32_t, N+1>& dimensions, size_t size) : 
		size_(size / dimensions.back()) 
	{
		memcpy(dimensions_.data(), dimensions.data() + 1, dimensions_.size() * sizeof(dimensions_[0]));
	}

	std::array<uint32_t, N> dimensions_;

	size_t size_;
};

template<>
class Shape<1>
{
public:
	Shape() :size_(0) {}

	Shape(size_t size) : size_(size) { assert(size_ != 0); }

	void operator = (size_t size) { size_ = size; }

	size_t size() const { return size_; }

	size_t length() const { return size_; }

	size_t length(uint32_t) const { return size_; }

protected:
	friend Shape<2>;

	Shape(const std::array<uint32_t, 2>& dimensions, size_t size) :
		size_(dimensions[1])
	{
	}

private:
	size_t size_;
};

template<>
class Shape<0>
{
	static const size_t capacity = 8;

public:
	Shape() : dimensionCount_(0), size_(0) {}

	template<size_t N, typename T> Shape(const T(&dimensions)[N])
	{
		static_assert(N < capacity, "Capacity of Shape<> too small.");

		dimensionCount_ = N;
		size_ = 1;
		for (size_t i = 0; i < dimensionCount_; ++i)
		{
			dimensions_[i] = dimensions[i];
			size_ *= dimensions_[i];
		}
		assert(size_ != 0);
	}

	Shape(const std::initializer_list<uint32_t>& dimensions)
	{
		assert(dimensions.size() < capacity);

		dimensionCount_ = dimensions.size();
		size_ = 1;
		uint32_t i = 0;
		for (auto dimension : dimensions)
		{
			dimensions_[i++] = dimension;
			size_ *= dimension;
		}
		assert(size_ != 0);
	}

	template<size_t N> Shape(const Shape<N>& shape)
	{
		static_assert(N < capacity, "Capacity of Shape<> too small.");

		dimensionCount_ = N;
		size_ = shape.size();
		memcpy(dimensions_.data(), shape.dimensions().data(), N * sizeof(dimensions_[0]));
	}

	void operator = (const std::initializer_list<uint32_t>& dimensions)
	{
		assert(dimensions.size() < capacity);

		dimensionCount_ = dimensions.size();
		size_ = 1;
		auto dimension = dimensions.begin();
		for (size_t i = 0; i < dimensionCount_; ++i)
		{
			dimensions_[i] = *dimension++;
			size_ *= dimensions_[i];
		}
	}

	size_t size() const { return size_; }

	const auto& dimensions() const { return dimensions_; }

	size_t length(uint32_t dim = 0) const
	{
		assert(dim < dimensionCount_);
		return dimensions()[dim];
	}

	auto slice() const
	{
		assert(dimensionCount_ > 1);
		return Shape(dimensions(), dimensionCount_ - 1, size_);
	}
private:
	Shape(const std::array<uint32_t, capacity>& dimensions, size_t dimensionCount, size_t size) :
		size_(size / dimensions.back()), dimensionCount_(dimensionCount)
	{
		memcpy(dimensions_.data(), dimensions.data() + 1, dimensionCount * sizeof(dimensions_[0]));
	}

	std::array<uint32_t, capacity> dimensions_;

	uint32_t dimensionCount_;

	size_t size_;
};

template<size_t N> inline bool operator==(const Shape<N>& lhs, const Shape<N>& rhs)
{
	return lhs.dimensions() == rhs.dimensions();
}

template<> inline bool operator==(const Shape<1>& lhs, const Shape<1>& rhs)
{
	return lhs.size() == rhs.size();
}

template<size_t N> inline bool operator==(const Shape<N>& lhs, const Shape<0>& rhs)
{
	if (N != rhs.dimensions().size()) return false;
	return std::memcmp(lhs.dimensions().data(), rhs.dimensions().data(), N * sizeof(rhs.dimensions()[0])) == 0;	
}

template<size_t N> inline bool operator==(const Shape<0>& lhs, const Shape<N>& rhs)
{
	return rhs == lhs;
}

inline bool operator==(const Shape<0>& lhs, const Shape<0>& rhs)
{
	if (rhs.dimensions().size() != rhs.dimensions().size()) return false;
	return std::memcmp(lhs.dimensions().data(), rhs.dimensions().data(), rhs.size() * sizeof(rhs.dimensions()[0])) == 0;
}

template<size_t N, size_t M> inline bool operator!=(const Shape<N>& lhs, const Shape<M>& rhs)
{
	return !(lhs == rhs);
}
}