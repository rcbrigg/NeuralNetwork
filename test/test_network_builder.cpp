#include "pch.h"
#include "CppUnitTest.h"

#include "..\include\network.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace nn;
using namespace std;

namespace test
{
	TEST_CLASS(NetworkBuilder)
	{
	public:
		
		TEST_METHOD(InvalidArgs)
		{
			auto invalidArgs = [] 
			{  
				NetworkArgs args;
				auto network = Network(move(args));
			};		
			Assert::ExpectException<std::invalid_argument>(invalidArgs);
		}

		TEST_METHOD(BadInputShape)
		{
			auto badInputShape = []
			{
				NetworkArgs args;
				args.setInputShape(Shape<3>());
			};
			Assert::ExpectException<std::invalid_argument>(badInputShape);
		}

		TEST_METHOD(InputShapeNotSet)
		{
			auto badInputShape = []
			{
				NetworkArgs args;
				args.addLayerSigmoid();
			};
			Assert::ExpectException<std::invalid_argument>(badInputShape);
		}

		TEST_METHOD(OutputShape)
		{
			NetworkArgs args;
			args.setInputShape({ 16, 10 });
			args.addLayerSigmoid();
			Assert::IsTrue(args.getOutputShape() == Shape<>({ 16, 10 }));
		}
	};
}
