#include "LinearRegression.h"
#include <random>
#include <omp.h>
#include <iostream>

LinearRegression::LinearRegression(int trainingSize, int numFeatures, double* trainingData, double* trainingLabels, 
                                double convergenceThreshold, double learningRate, int maxIterations)
:
    m_ConvergenceThreshold(convergenceThreshold), 
    m_LearningRate(learningRate),
    m_TrainingSize(trainingSize),
    m_NumFeatures(numFeatures),
    m_TrainingData(trainingData),
    m_TrainingLabels(trainingLabels),
    m_MaxIterations(maxIterations)
{
    m_Theta = new double[m_NumFeatures + 1];
    m_LastPredictions = new double[m_TrainingSize];
}

LinearRegression::~LinearRegression()
{
    delete[] m_Theta;
    delete[] m_LastPredictions;
}

void LinearRegression::initializeTheta()
{
    for (int j = 0; j < m_NumFeatures; ++j)
    {
        m_Theta[j] = 0;
    }
    m_Theta[m_NumFeatures] = 0;
}

void LinearRegression::fit()
{
    int iteration = 0;
    initializeTheta();
    double error = mse();
    std::cout << "Initial MSE: " << error << std::endl;
    while (error > m_ConvergenceThreshold && iteration < m_MaxIterations)
    {
        updateTheta();
        error = mse();
        ++iteration;
    }
    std::cout << "Final MSE: " << error << std::endl;
}

double LinearRegression::predict(double* x)
{
    double prediction = 0;
    for (int j = 0; j < m_NumFeatures; ++j)
    {
        prediction += m_Theta[j] * x[j];
    }
    prediction += m_Theta[m_NumFeatures];
    return prediction;
}

void LinearRegression::updateTheta()
{
    double* gradient = computeGradient();
    for (int j = 0; j < m_NumFeatures; ++j)
    {
        m_Theta[j] -= m_LearningRate * gradient[j];
    }
    m_Theta[m_NumFeatures] -= m_LearningRate * gradient[m_NumFeatures];
    delete[] gradient;
}

double LinearRegression::mse()
{
    double error = 0;
    #pragma omp parallel for reduction(+:error) schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < m_TrainingSize; ++i)
    {
        double prediction = predict(m_TrainingData + i * m_NumFeatures);
        error += (prediction - m_TrainingLabels[i]) * (prediction - m_TrainingLabels[i]);
        m_LastPredictions[i] = prediction;
    }
    return error / m_TrainingSize;
}

double* LinearRegression::computeGradient()
{
    double* gradient = new double[m_NumFeatures + 1];
    std::fill(gradient, gradient + m_NumFeatures + 1, 0);
    
    int chunkSize = std::ceil(static_cast<double>(m_TrainingSize) / static_cast<double>(NUM_THREADS));

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int id = omp_get_thread_num();
        int myStart = id * chunkSize;
        int myEnd = std::min(myStart + chunkSize, m_TrainingSize);
        double* myGradient = new double[m_NumFeatures + 1];
        std::fill(myGradient, myGradient + m_NumFeatures + 1, 0);

        for (int i = myStart; i < myEnd; ++i)
        {
            double difference = m_LastPredictions[i] - m_TrainingLabels[i];
            for (int j = 0; j < m_NumFeatures; ++j)
            {
                myGradient[j] += difference * m_TrainingData[i * m_NumFeatures + j];
            }
            myGradient[m_NumFeatures] += difference;
        }
        for (int j = 0; j < m_NumFeatures; ++j)
        {
            #pragma omp atomic update
            gradient[j] += myGradient[j];
        }
        #pragma omp atomic update
        gradient[m_NumFeatures] += myGradient[m_NumFeatures];
        delete[] myGradient;
    }

    for (int j = 0; j < m_NumFeatures; ++j)
    {
        gradient[j] *= (double(2) / m_TrainingSize);
    }
    gradient[m_NumFeatures] *= (double(2) / m_TrainingSize);
 
    return gradient;
}
