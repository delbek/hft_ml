class LinearRegression 
{
public:
    LinearRegression(int trainingSize, int numFeatures, float* trainingData, float* trainingLabels, float convergenceThreshold, float learningRate);
    LinearRegression(LinearRegression&& other) = delete;
    LinearRegression& operator=(LinearRegression&& other) = delete; 
    LinearRegression(const LinearRegression& other) = delete;
    LinearRegression& operator=(const LinearRegression& other) = delete;
    ~LinearRegression();

    void fit();
    float predict(float* x);

private:
    void updateTheta();
    void initializeTheta();
    float* computeGradient();
    float mse();

private:
    int m_TrainingSize;
    int m_NumFeatures;
    float* m_TrainingData; // of size trainingSize * numFeatures
    float* m_TrainingLabels; // of size trainingSize
    float m_ConvergenceThreshold;
    float m_LearningRate;
    float* m_Theta;
    float* m_LastPredictions;
};
