#include "PolynomialFeatures.h"
#include <cmath>

PolynomialFeatures::PolynomialFeatures(int trainingSize, int numFeatures, double* trainingData, int degree)
:
    m_TrainingSize(trainingSize),
    m_NumFeatures(numFeatures),
    m_TrainingData(trainingData),
    m_Degree(degree)
{}

double* PolynomialFeatures::transform()
{
    double* transformedData = new double[m_TrainingSize * m_NumFeatures * m_Degree];
    #pragma omp parallel for collapse(3) schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < m_TrainingSize; ++i)
    {
        for (int j = 0; j < m_NumFeatures; ++j)
        {
            for (int k = 1; k <= m_Degree; ++k)
            {
                transformedData[i * m_NumFeatures * m_Degree + j * m_Degree + k] = pow(m_TrainingData[i * m_NumFeatures + j], k);
            }
        }
    }
    return transformedData;
}
