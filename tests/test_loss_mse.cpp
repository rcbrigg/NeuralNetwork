#include "pch.h"
#include "..\src\losses\mse.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace test
{
namespace loss
{
TEST_CLASS(Mse)
{
public:

	TEST_METHOD(Error)
	{
		float output[] = { 0.f, 1.f, 2.0f };
		float target[] = { -1.f, 1.f, 5.f };
		float expected[] = { 1.f, 0.f, 9.f };
		float error[3];

		nn::loss::Mse lossFunc;
		lossFunc.calculateError(output, target, error, 3);
		for (size_t i = 0; i < 3; ++i)
		{
			Assert::AreEqual(expected[i], error[i]);
		}
		
		auto loss = lossFunc.calculateError(output, target, 3);
		Assert::AreEqual(10.f, loss);
	}
};
}
}