#include "config.h"

class PolynomialFeatures
{
public:
    PolynomialFeatures(int trainingSize, int numFeatures, double* trainingData, int degree);
    PolynomialFeatures(const PolynomialFeatures& other) = delete;
    PolynomialFeatures& operator=(const PolynomialFeatures& other) = delete;
    PolynomialFeatures(PolynomialFeatures&& other) noexcept = delete;
    PolynomialFeatures& operator=(PolynomialFeatures&& other) noexcept = delete;
    ~PolynomialFeatures() = default;
    double* transform();

private:
    int m_TrainingSize;
    int m_NumFeatures;
    double* m_TrainingData;
    int m_Degree;
};
