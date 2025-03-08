#include "LinearRegression.h"
#include <iostream>

int main()
{
    float* trainingData = new float[10 * 1];
    float* trainingLabels = new float[10];
    for (int i = 0; i < 10; ++i)
    {
        trainingData[i] = i;
        trainingLabels[i] = i;
    }
    LinearRegression lr(10, 1, trainingData, trainingLabels, 0.001, 0.01);
    lr.fit();
    float* testData = new float[10 * 1];
    for (int i = 0; i < 10; ++i)
    {
        testData[i] = i * 5;
    }
    float* predictions = new float[10];
    for (int i = 0; i < 10; ++i)
    {
        predictions[i] = lr.predict(testData + i);
    }
    for (int i = 0; i < 10; ++i)
    {
        std::cout << predictions[i] << std::endl;
    }

    delete[] trainingData;
    delete[] trainingLabels;
    delete[] testData;
    delete[] predictions;

    return 0;
}
