#pragma once
#include <stdint.h>
#include "shape.hpp"
#include <cassert>
#include <utility>
#include <memory>

namespace nn
{

template<size_t N, typename Type>
class TensorBase
{
public:
	using value_type = Type;

	size_t size() const { return shape().size(); }

	size_t length(size_t dimension = 0) const { return shape().length(dimension); }

	const Shape<N>& shape() const { return shape_; }

	value_type* data() { return data_; }

	const value_type* data() const { return data_; }

	value_type* end() { return data() + size(); }

	const value_type* end() const { return data() + size(); }

	value_type* data(ptrdiff_t offset)
	{
		return (offset >= 0 ? data() : end()) + offset;
	}

	value_type* data(ptrdiff_t offset) const
	{
		return (offset >= 0 ? data() : end()) + offset;
	}

	template<typename T> bool operator == (const TensorBase<N, T>& rhs) const
	{
		if (shape() != rhs.shape()) return false;
		return memcmp(data(), rhs.data(), size() * sizeof(float)) == 0;
	}

protected:
	TensorBase() : data_(nullptr) {}

	TensorBase(Shape<N>&& shape, Type* data) :
		shape_(std::move(shape)), data_(data)
	{
	}

	TensorBase(Shape<N>&& shape, const std::shared_ptr<Type>& data) :
		shape_(std::move(shape)), selfManaged_(data), data_(data.get())
	{
	}

	TensorBase(Shape<N>&& shape) :
		shape_(std::move(shape)),
		selfManaged_(new Type[shape.size()], std::default_delete<Type[]>()),
		data_(selfManaged_.get())
	{
	}

	Shape<N> shape_;

	std::shared_ptr<Type> selfManaged_;

	Type* data_;
};

template<size_t N = 1, typename Type = float>
class Tensor : public TensorBase<N, Type>
{
	using Base = TensorBase<N, Type>;

public:
	using value_type = Type;

	Tensor() : Base() {}

	Tensor(Shape<N> shape, Type* data) : Base(move(shape), data)
	{
	}

	Tensor(Shape<N> shape, const std::shared_ptr<Type>& data) : Base(move(shape), data)
	{
	}

	Tensor(Shape<N> shape) : Base(std::move(shape))
	{
	}

	Tensor(const std::initializer_list<size_t>& dimensions) :
		Tensor(Shape<N>(dimensions))
	{
	}

	auto flat() { return as<1>({ Base::size() }); }

	auto flat() const { return as<1>({ Base::size() }); }

	template<size_t M> auto as(Shape<M> newShape)
	{
		assert(newShape.size() == Base::size());
		if (Base::selfManaged_)
		{
			return Tensor<M, value_type>(newShape, Base::selfManaged_);
		}
		else
		{
			return Tensor<M, value_type>(newShape, Base::data_);
		}
	}

	template<size_t M> auto as(Shape<M> newShape) const
	{
		assert(newShape.size() == Base::size());
		if (Base::selfManaged_)
		{
			return Tensor<M, const value_type>(newShape, Base::selfManaged_);
		}
		else
		{
			return Tensor<M, const value_type>(newShape, Base::data_);
		}
	}

	auto asConst() const
	{
		Tensor<N, const value_type> t;
		t.data_ = Base::data_;
		t.shape_ = Base::shape_;
		t.selfManaged_ = Base::selfManaged_;
		return t;
	}

	auto operator [] (size_t i)
	{
		assert(i < Base::length());
		Shape<N-1> newShape = Base::shape().slice();
		return Tensor<N - 1, value_type>(newShape, Base::data() + i * newShape.size());
	}

	auto operator [] (size_t i) const
	{
		assert(i < Base::length());
		Shape newShape = Base::shape().slice();
		return Tensor<N - 1, const value_type>(newShape, Base::data() + i * newShape.size());
	}

	Tensor<N, value_type> section(size_t first, size_t last)
	{
		assert(last <= Base::size() && first < last);
		Tensor<N, value_type> t;
		auto dimensions = Base::shape_.dimensions();
		dimensions[0] = last - first;
		t.shape_ = Shape<N>(dimensions);
		t.data_ = Base::data_ + first * (Base::size() / Base::length());
		if (Base::selfManaged_)
		{
			t.selfManaged_ = std::shared_ptr<Type>(Base::selfManaged_, t.data_);
		}
		return t;
	}
};

template<typename Type>
class Tensor<1, Type> : public TensorBase<1, Type>
{
	using Base = TensorBase<1, Type>;

public:
	using value_type = Type;

	Tensor() : Base() {}

	Tensor(Shape<1> shape, Type* data) : Base(std::move(shape), data)
	{
	}

	Tensor(Shape<1> shape, const std::shared_ptr<Type>& data) : Base(std::move(shape), data)
	{
	}

	Tensor(Shape<1> shape) : Base(std::move(shape))
	{
	}

	Tensor(const std::initializer_list<Type>& data) : Base(Shape<1>(data.size()))
	{
		std::copy(data.begin(), data.end(), Base::data());
	}

	auto flat() { return as<1>({ Base::size() }); }

	auto flat() const { return as<1>({ Base::size() }); }

	template<size_t M> auto as(Shape<M> newShape)
	{
		assert(newShape.size() == Base::size());
		if (Base::selfManaged_)
		{
			return Tensor<M, value_type>(newShape, Base::selfManaged_);
		}
		else
		{
			return Tensor<M, value_type>(newShape, Base::data_);
		}
	}

	template<size_t M> auto as(Shape<M> newShape) const
	{
		assert(newShape.size() == Base::size());
		if (Base::selfManaged_)
		{
			return Tensor<M, const value_type>(newShape, Base::selfManaged_);
		}
		else
		{
			return Tensor<M, const value_type>(newShape, Base::data_);
		}
	}

	auto& operator [] (size_t i)
	{
		assert(i < Base::length());
		return Base::data()[i];
	}

	auto& operator [] (size_t i) const
	{
		assert(i < Base::length());
		return Base::data()[i];
	}

	Tensor<1, value_type> section(size_t first, size_t last)
	{
		assert(last <= Base::size() && first < last);
		Tensor<1, value_type> t;
		t.shape_ = Shape<1>(last - first);
		t.data_ = Base::data_ + first;
		if (Base::selfManaged_)
		{
			t.selfManaged_ = std::shared_ptr< Type>(Base::selfManaged_, t.data_);
		}
		return t;
	}
};

template <size_t N = 1>
using ConstTensor = Tensor<N, const float>;
}
