#pragma once
#include <stdint.h>
#include "shape.hpp"
#include <cassert>
#include <utility>

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

	~TensorBase()
	{
		if (owner_)
		{
			delete[] data_;
		}
	}

protected:
	TensorBase() : data_(nullptr), owner_(false) {}

	TensorBase(Shape<N>&& shape, Type* data) :
		owner_(false), shape_(move(shape)), data_(data)
	{
	}

	TensorBase(Shape<N>&& shape) :
		owner_(true), shape_(std::move(shape)), data_(new Type[shape.size()])
	{
	}

private:
	bool owner_;

	Shape<N> shape_;

	Type* data_;
};

template<size_t N = 1, typename Type = float>
class Tensor : public TensorBase<N, Type>
{
	using Base = TensorBase<N, Type>;

public:
	using value_type = Type;

	Tensor() : TensorBase() {}

	Tensor(Shape<N> shape, Type* data) : TensorBase(move(shape), data)
	{
	}

	Tensor(Shape<N> shape) : Base(std::move(shape))
	{
	}

	template<size_t M> auto as(Shape<M> newShape)
	{
		assert(newShape.size() == Base::size());
		return Tensor<M, value_type>(newShape, TensorBase::data());
	}

	template<size_t M> auto as(Shape<M> newShape) const
	{
		assert(newShape.size() == Base::size());
		return Tensor<M, const value_type>(newShape, Base::data());
	}

	auto flat() { return as<1>({ Base::size() }); }

	auto flat() const { return as<1>({ Base::size() }); }

	auto operator [] (size_t i)
	{
		assert(i < Base::length());
		Shape newShape = Base::shape().slice();
		return Tensor<N - 1, value_type>(newShape, Base::data() + i * newShape.size());
	}

	auto operator [] (size_t i) const
	{
		assert(i < Base::length());
		Shape newShape = Base::shape().slice();
		return Tensor<N - 1, const value_type>(newShape, Base::data() + i * newShape.size());
	}
};

template<typename Type>
class Tensor<1, Type> : public TensorBase<1, Type>
{
	using Base = TensorBase<1, Type>;

public:
	using value_type = Type;

	Tensor() : Base() {}

	Tensor(Shape<1> shape, Type* data) : TensorBase(std::move(shape), data_(data))
	{
	}

	Tensor(Shape<1> shape) : Base(std::move(shape))
	{
	}

	Tensor(const std::initializer_list<Type>& data) : Base(Shape<1>(data.size()))
	{
		std::copy(data.begin(), data.end(), Base::data());
	}

	template<size_t M> auto as(Shape<M> newShape)
	{
		assert(newShape.size() == Base::size());
		return Tensor<M, value_type>(newShape, Base::data());
	}

	template<size_t M> auto as(Shape<M> newShape) const
	{
		assert(newShape.size() == Base::size());
		return Tensor<M, const value_type>(newShape, Base::data());
	}

	auto operator [] (size_t i)
	{
		assert(i < TensorBase::length());
		return Base::data()[i];
	}

	auto operator [] (size_t i) const
	{
		assert(i < TensorBase::length());
		return Base::data()[i];
	}
};

template <size_t N = 1>
using ConstTensor = Tensor<N, const float>;
}
