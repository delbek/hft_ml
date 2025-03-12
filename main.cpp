#include "LinearRegression.h"
#include "PolynomialFeatures.h"
#include <cmath>
#include <iostream>


void generateLinearData(double*& trainingData, double*& trainingLabels, int dataSize, int featureSize)
{
    // y = x + 3
    trainingData = new double[dataSize * featureSize];
    trainingLabels = new double[dataSize];
    for (int i = 0; i < dataSize; ++i)
    {
        for (int j = 0; j < featureSize; ++j)
        {
            trainingData[i * featureSize + j] = i;
        }
        trainingLabels[i] = i + 3;
    }
}

void generatePolynomialData(double*& trainingData, double*& trainingLabels, int dataSize, int featureSize, int degree)
{
    // y = x^2 + 3
    trainingData = new double[dataSize * featureSize * degree];
    trainingLabels = new double[dataSize];
    for (int i = 0; i < dataSize; ++i)
    {
        for (int j = 0; j < featureSize; ++j)
        {
            trainingData[i * featureSize * degree + j * degree] = i;
        }
        trainingLabels[i] = pow(i, 2) + 3;
    }
}

double computeMSE(double* predictions, double* labels, int size)
{
    double mse = 0;
    for (int i = 0; i < size; ++i)
    {
        mse += pow(predictions[i] - labels[i], 2);
    }
    return mse / size;
}

#define DATA_SIZE 10
#define FEATURE_SIZE 2
#define DEGREE 2

int main()
{
    double* trainingData;
    double* trainingLabels;
    generatePolynomialData(trainingData, trainingLabels, DATA_SIZE, FEATURE_SIZE, DEGREE);
    
    PolynomialFeatures polynomialFeatures(DATA_SIZE, FEATURE_SIZE, trainingData, DEGREE);
    double* transformedData = polynomialFeatures.transform();

    LinearRegression lr(DATA_SIZE, FEATURE_SIZE * DEGREE, transformedData, trainingLabels);
    lr.fit();

    double* testData;
    double* testLabels;
    generatePolynomialData(testData, testLabels, DATA_SIZE, FEATURE_SIZE, DEGREE);

    double* predictions = new double[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; ++i)
    {
        predictions[i] = lr.predict(testData + i * FEATURE_SIZE * DEGREE);
    }

    double mse = computeMSE(predictions, trainingLabels, DATA_SIZE);
    std::cout << "Predicted MSE: " << mse << std::endl;

    delete[] trainingData;
    delete[] trainingLabels;
    delete[] testData;
    delete[] testLabels;
    delete[] predictions;

    return 0;
}
