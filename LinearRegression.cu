#include "LinearRegression.h"
#include <random>
#include <omp.h>

LinearRegression::LinearRegression(int trainingSize, int numFeatures, float* trainingData, float* trainingLabels, float convergenceThreshold, float learningRate)
:
    m_ConvergenceThreshold(convergenceThreshold), 
    m_LearningRate(learningRate),
    m_TrainingSize(trainingSize),
    m_NumFeatures(numFeatures),
    m_TrainingData(trainingData),
    m_TrainingLabels(trainingLabels)
{
    m_Theta = new float[numFeatures];
    m_LastPredictions = new float[trainingSize];
}

LinearRegression::~LinearRegression()
{
    delete[] m_Theta;
    delete[] m_LastPredictions;
}

void LinearRegression::initializeTheta()
{
    for (int i = 0; i < m_NumFeatures; ++i)
    {
        m_Theta[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void LinearRegression::fit()
{
    initializeTheta();
    float error = mse();
    while (error > m_ConvergenceThreshold)
    {
        updateTheta();
        error = mse();
    }
}

float LinearRegression::predict(float* x)
{
    float prediction = 0.0f;
    for (int i = 0; i < m_NumFeatures; ++i)
    {
        prediction += m_Theta[i] * x[i];
    }
    return prediction;
}

void LinearRegression::updateTheta()
{
    float* gradient = computeGradient();
    for (int i = 0; i < m_NumFeatures; ++i)
    {
        m_Theta[i] -= m_LearningRate * gradient[i];
    }
    delete[] gradient;
}

float LinearRegression::mse()
{
    float error = 0.0f;
    #pragma omp parallel for reduction(+:error) schedule(static) num_threads(omp_get_max_threads())
    for (int i = 0; i < m_TrainingSize; ++i)
    {
        float prediction = predict(m_TrainingData + i * m_NumFeatures);
        error += (prediction - m_TrainingLabels[i]) * (prediction - m_TrainingLabels[i]);
        m_LastPredictions[i] = prediction;
    }
    return error / m_TrainingSize;
}

float* LinearRegression::computeGradient()
{
    float* gradient = new float[m_NumFeatures];
    std::fill(gradient, gradient + m_NumFeatures, 0.0f);
    
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        int id = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int myStart = id * (m_TrainingSize / numThreads);
        int myEnd = myStart + (m_TrainingSize / numThreads);
        float* myGradient = new float[m_NumFeatures];

        for (int i = myStart; i < myEnd; ++i)
        {
            for (int j = 0; j < m_NumFeatures; ++j)
            {
                myGradient[j] += (m_LastPredictions[i] - m_TrainingLabels[i]) * m_TrainingData[i * m_NumFeatures + j];
            }
        }
        for (int j = 0; j < m_NumFeatures; ++j)
        {
            #pragma omp atomic update
            gradient[j] += myGradient[j];
        }
        delete[] myGradient;
    }

    for (int j = 0; j < m_NumFeatures; ++j)
    {
        gradient[j] *= (2.0f / m_TrainingSize);
    }
 
    return gradient;
}
