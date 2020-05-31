#include "..\include\tensor.hpp"
#include <fstream>
#include <assert.h>

struct MnistData
{
	nn::Tensor<3> trainingData;
	nn::Tensor<3> testData;
	nn::Tensor<1, uint32_t> trainingLabels;
	nn::Tensor<1, uint32_t> testLabels;
};

static uint32_t readInteger(std::ifstream& file)
{
	char bytes[4];
	file.read(bytes, 4);
	std::swap(bytes[0], bytes[3]);
	std::swap(bytes[1], bytes[2]);
	return *(uint32_t*)(bytes);
}

nn::Tensor<3> loadInput(std::string path)
{
	path = "../../data/" + path;
	auto file = std::ifstream(path, std::ios::binary);
	if (file.fail())
	{
		throw std::exception();
	}
	
	auto magicNumber = readInteger(file);
	assert(magicNumber == 0x0803);
	auto count = readInteger(file);
	auto height = readInteger(file);
	auto width = readInteger(file);

	auto result = nn::Tensor<3>({ count, height, width });
	auto buffer = new uint8_t[result.size()];
	file.read((char*)buffer, result.size());

	for (size_t i = 0; i < result.size(); ++i)
	{
		result.data()[i] = (float)buffer[i] / 256.f;
	}
	delete[] buffer;

	return result;
}

nn::Tensor<1, uint32_t> loadLabels(std::string path)
{
	path = "../../data/" + path;
	auto file = std::ifstream(path, std::ios::binary);
	if (file.fail())
	{
		throw std::exception();
	}

	auto magicNumber = readInteger(file);
	assert(magicNumber == 0x0801);
	auto count = readInteger(file);

	// Unfornatley we only support labels as 32 bit unsigned ints
	// so we have to convert each element
	auto result = nn::Tensor<1, uint32_t>(count);
	auto buffer = new uint8_t[count];
	file.read((char*)buffer, result.size() * sizeof(uint8_t));
	for (size_t i = 0; i < result.size(); ++i)
	{
		result[i] = buffer[i];
	}
	delete[] buffer;

	return result;
}

MnistData loadMnistData()
{
	MnistData data;
	data.trainingData = loadInput("train-images.idx3-ubyte");
	data.testData = loadInput("t10k-images.idx3-ubyte");
	data.trainingLabels = loadLabels("train-labels.idx1-ubyte");
	data.testLabels = loadLabels("t10k-labels.idx1-ubyte");
	return data;
}

void readTensor(nn::Tensor<3, float>& tensor, std::ifstream& file)
{
	uint32_t size;
	file >> size;
	tensor = nn::Tensor<3, float>({ size, 28, 28 });
	file.read((char*)tensor.data(), tensor.size() * sizeof(float));
};

void readTensor(nn::Tensor<1, uint32_t>& tensor, std::ifstream& file)
{
	uint32_t size;
	file >> size;
	tensor = nn::Tensor<1, uint32_t>(size);
	file.read((char*)tensor.data(), tensor.size() * sizeof(uint32_t));
};

template<typename T, size_t N> void writeTensor(const nn::Tensor<N, T>& tensor, std::ofstream& file)
{
	uint32_t size = tensor.length();
	file << size;
	file.write((char*)tensor.data(), tensor.size() * sizeof(T));
};

MnistData LoadFormattedMnist()
{
	auto file = std::ifstream("../../data/mnist.bin", std::ios::binary);

	if (file.fail())
	{
		throw std::exception();
	}

	MnistData data;
	readTensor(data.trainingData, file);
	readTensor(data.testData, file);
	readTensor(data.trainingLabels, file);
	readTensor(data.testLabels, file);
	return data;
}

void DumpFormattedMnist(const MnistData& data)
{
	auto file = std::ofstream("../../data/mnist.bin", std::ios::binary);

	writeTensor(data.trainingData, file);
	writeTensor(data.testData, file);
	writeTensor(data.trainingLabels, file);
	writeTensor(data.testLabels, file);
}
