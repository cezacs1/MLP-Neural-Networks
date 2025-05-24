using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

public class NeuronModelData
{
    public double[] Weights { get; set; }
    public double Bias { get; set; }
}

public class LayerModelData
{
    public string ActivationFunctionName { get; set; }
    public bool IsOutputLayerWithSoftmax { get; set; }
    public double DropoutRate { get; set; }
    public List<NeuronModelData> Neurons { get; set; }
}

public class NeuralNetworkModelData
{
    public int InputWindowSize { get; set; }
    public int VocabularySize { get; set; }
    public int[] HiddenLayerSizes { get; set; }
    public int OutputSize { get; set; }
    public double LearningRate { get; set; }
    public List<LayerModelData> LayersData { get; set; }
}

public class VocabularyModelData
{
    public Dictionary<string, int> WordToIndex { get; set; }
    public Dictionary<int, string> IndexToWord { get; set; }
    public int CurrentIndex { get; set; }
}

public class Vocabulary
{
    private Dictionary<string, int> _wordToIndex;
    private Dictionary<int, string> _indexToWord;
    private int _currentIndex;
    public const string PadToken = "<PAD>";
    public const string UnknownToken = "<UNK>";

    public int Size => _wordToIndex.Count;

    public Vocabulary()
    {
        _wordToIndex = new Dictionary<string, int>();
        _indexToWord = new Dictionary<int, string>();
        _currentIndex = 0;
        AddWord(PadToken);
        AddWord(UnknownToken);
    }

    public void AddWord(string word)
    {
        if (string.IsNullOrEmpty(word)) return;
        if (!_wordToIndex.ContainsKey(word))
        {
            _wordToIndex[word] = _currentIndex;
            _indexToWord[_currentIndex] = word;
            _currentIndex++;
        }
    }

    public int GetIndex(string word)
    {
        if (string.IsNullOrEmpty(word)) return _wordToIndex[PadToken];
        return _wordToIndex.TryGetValue(word, out int index) ? index : _wordToIndex[UnknownToken];
    }

    public string GetWord(int index)
    {
        return _indexToWord.TryGetValue(index, out string word) ? word : UnknownToken;
    }

    public void Build(IEnumerable<string[]> corpus)
    {
        foreach (var sequence in corpus)
        {
            foreach (var word in sequence)
            {
                AddWord(word);
            }
        }
    }

    public double[] GetOneHotVector(string word)
    {
        int index = GetIndex(word);
        double[] vector = new double[Size];
        if (index >= 0 && index < Size)
        {
            vector[index] = 1.0;
        }
        else { vector[_wordToIndex[UnknownToken]] = 1.0; }
        return vector;
    }
    public double[] GetOneHotVectorByIndex(int index)
    {
        double[] vector = new double[Size];
        if (index >= 0 && index < Size)
        {
            vector[index] = 1.0;
        }
        else { vector[_wordToIndex[UnknownToken]] = 1.0; }
        return vector;
    }

    public void Save(string filePath)
    {
        var modelData = new VocabularyModelData
        {
            WordToIndex = this._wordToIndex,
            IndexToWord = this._indexToWord,
            CurrentIndex = this._currentIndex
        };
        string json = JsonConvert.SerializeObject(modelData, Formatting.Indented);
        File.WriteAllText(filePath, json);
        Console.WriteLine($"Kelime dağarcığı kaydedildi: {filePath}");
    }

    public static Vocabulary Load(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Kelime dağarcığı dosyası bulunamadı: {filePath}");
            return null;
        }
        string json = File.ReadAllText(filePath);
        var modelData = JsonConvert.DeserializeObject<VocabularyModelData>(json);

        var vocab = new Vocabulary();
        vocab._wordToIndex = modelData.WordToIndex;
        vocab._indexToWord = modelData.IndexToWord;
        vocab._currentIndex = modelData.CurrentIndex;
        Console.WriteLine($"Kelime dağarcığı yüklendi: {filePath}");
        return vocab;
    }
}

public static class ActivationFunctions
{
    public static double Sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }
    public static double SigmoidDerivative(double sigmoidOutput) { return sigmoidOutput * (1.0 - sigmoidOutput); }
    public static double Identity(double x) { return x; }
    public static double IdentityDerivative(double x) { return 1.0; }
    public static double[] Softmax(double[] logits)
    {
        if (logits == null || logits.Length == 0) return new double[0];
        double maxLogit = logits.Max();
        double[] expValues = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
        double sumExpValues = expValues.Sum();
        if (sumExpValues == 0) sumExpValues = 1e-9;
        return expValues.Select(x => x / sumExpValues).ToArray();
    }
    public static Func<double, double> GetActivationFunction(string name)
    {
        switch (name?.ToLower())
        {
            case "sigmoid": return Sigmoid;
            case "identity": return Identity;
            default: throw new ArgumentException($"Bilinmeyen aktivasyon fonksiyonu: {name}");
        }
    }
    public static Func<double, double> GetActivationDerivative(string name)
    {
        switch (name?.ToLower())
        {
            case "sigmoid": return SigmoidDerivative;
            case "identity": return IdentityDerivative;
            default: throw new ArgumentException($"Bilinmeyen aktivasyon türevi: {name}");
        }
    }
}

public class Neuron
{
    public double[] Weights { get; set; }
    public double Bias { get; set; }
    [JsonIgnore] public double Output { get; set; }
    [JsonIgnore] public double InputToActivation { get; set; }
    [JsonIgnore] public double Delta { get; set; }
    public Neuron() { }
    public Neuron(int inputCount, Random random)
    {
        Bias = (random.NextDouble() * 2.0 - 1.0) * 0.1;
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
        {
            Weights[i] = (random.NextDouble() * 2.0 - 1.0) * Math.Sqrt(1.0 / inputCount);
        }
    }
}

public class Layer
{
    public List<Neuron> Neurons { get; set; }
    public string ActivationFunctionName { get; set; }
    public bool IsOutputLayerWithSoftmax { get; set; }
    public double DropoutRate { get; set; }

    [JsonIgnore] public Func<double, double> ActivationFunction { get; private set; }
    [JsonIgnore] public Func<double, double> ActivationDerivative { get; private set; }

    [JsonIgnore] private double[] _lastDropoutMask;
    [JsonIgnore] private double[] _outputsBeforeDropout;
    [JsonIgnore] public double[] LastInputs { get; private set; }

    public Layer() { Neurons = new List<Neuron>(); }

    public Layer(int neuronCount, int inputCountPerNeuron, Random random,
                 string activationFuncName, double dropoutRate = 0.0, bool isOutputLayerWithSoftmax = false)
    {
        Neurons = new List<Neuron>();
        this.ActivationFunctionName = activationFuncName;
        AssignActivationFunctions();
        this.IsOutputLayerWithSoftmax = isOutputLayerWithSoftmax;
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate), "Dropout oranı [0, 1) aralığında olmalıdır.");
        this.DropoutRate = dropoutRate;

        for (int i = 0; i < neuronCount; i++)
        {
            Neurons.Add(new Neuron(inputCountPerNeuron, random));
        }
    }

    public void AssignActivationFunctions()
    {
        ActivationFunction = ActivationFunctions.GetActivationFunction(this.ActivationFunctionName);
        ActivationDerivative = ActivationFunctions.GetActivationDerivative(this.ActivationFunctionName);
    }

    public double[] CalculateOutputs(double[] inputs, bool isTraining, Random random)
    {
        this.LastInputs = inputs;
        double[] currentOutputs = new double[Neurons.Count];
        _outputsBeforeDropout = new double[Neurons.Count];

        for (int i = 0; i < Neurons.Count; i++)
        {
            Neuron neuron = Neurons[i];
            neuron.InputToActivation = neuron.Bias;
            for (int j = 0; j < neuron.Weights.Length; j++)
            {
                neuron.InputToActivation += neuron.Weights[j] * inputs[j];
            }

            if (IsOutputLayerWithSoftmax)
            {
                neuron.Output = neuron.InputToActivation;
                _outputsBeforeDropout[i] = neuron.Output;
                currentOutputs[i] = neuron.Output;
            }
            else
            {
                neuron.Output = ActivationFunction(neuron.InputToActivation);
                _outputsBeforeDropout[i] = neuron.Output;
                currentOutputs[i] = neuron.Output;
            }
        }

        if (!IsOutputLayerWithSoftmax && isTraining && DropoutRate > 0)
        {
            _lastDropoutMask = new double[Neurons.Count];
            double scaleFactor = 1.0 / (1.0 - DropoutRate);
            for (int i = 0; i < Neurons.Count; i++)
            {
                if (random.NextDouble() < DropoutRate)
                {
                    currentOutputs[i] = 0.0;
                    _lastDropoutMask[i] = 0.0;
                }
                else
                {
                    currentOutputs[i] *= scaleFactor;
                    _lastDropoutMask[i] = scaleFactor;
                }
            }
        }
        else
        {
            _lastDropoutMask = null;
        }
        return currentOutputs;
    }

    public double[] GetLastDropoutMask()
    {
        return _lastDropoutMask;
    }

    public double[] GetOutputsBeforeDropout()
    {
        return _outputsBeforeDropout;
    }
}

public class NeuralNetwork
{
    private List<Layer> _mlpLayers;
    private double _learningRate;
    private Random _random;

    private int _inputWindowSize;
    private int _vocabularySize;
    private int[] _hiddenLayerSizes;
    private int _outputVocabularySize;

    private bool _isTraining = false;

    [JsonIgnore] private List<double[]> _mlpLayerInputsCache;

    private NeuralNetwork() { _mlpLayers = new List<Layer>(); _random = new Random(); }

    public NeuralNetwork(int inputWindowSize, int vocabularySize,
                         int[] hiddenLayerSizes, double learningRate, Random randomSeed = null)
    {
        this._inputWindowSize = inputWindowSize;
        this._vocabularySize = vocabularySize;
        this._hiddenLayerSizes = (int[])hiddenLayerSizes.Clone();
        this._outputVocabularySize = vocabularySize;
        this._learningRate = learningRate;
        this._random = randomSeed ?? new Random();

        _mlpLayers = new List<Layer>();
        int currentMlpInputSize = inputWindowSize * vocabularySize;

        for (int i = 0; i < hiddenLayerSizes.Length; i++)
        {
            double dropoutForThisLayer = 0.0;
            _mlpLayers.Add(new Layer(hiddenLayerSizes[i], currentMlpInputSize, _random, "sigmoid", dropoutForThisLayer, false));
            currentMlpInputSize = hiddenLayerSizes[i];
        }

        _mlpLayers.Add(new Layer(vocabularySize, currentMlpInputSize, _random, "identity", 0.0, true));
    }

    public void SetTrainingMode(bool isTraining)
    {
        this._isTraining = isTraining;
    }

    public double[] FeedForward(double[] concatenatedOneHotInput)
    {
        _mlpLayerInputsCache = new List<double[]>();
        _mlpLayerInputsCache.Add(concatenatedOneHotInput.ToArray());

        double[] currentLayerProcessingOutput = concatenatedOneHotInput;

        for (int i = 0; i < _mlpLayers.Count; i++)
        {
            Layer layer = _mlpLayers[i];
            if (layer.ActivationFunction == null) layer.AssignActivationFunctions();

            currentLayerProcessingOutput = layer.CalculateOutputs(currentLayerProcessingOutput, _isTraining, _random);

            if (layer.IsOutputLayerWithSoftmax)
            {
                double[] softmaxOutputs = ActivationFunctions.Softmax(currentLayerProcessingOutput);
                for (int k = 0; k < layer.Neurons.Count; ++k)
                {
                    layer.Neurons[k].Output = softmaxOutputs[k];
                }
                currentLayerProcessingOutput = softmaxOutputs;
            }

            if (i < _mlpLayers.Count - 1)
            {
                _mlpLayerInputsCache.Add(currentLayerProcessingOutput.ToArray());
            }
        }
        return currentLayerProcessingOutput;
    }

    public void Backpropagate(double[] concatenatedOneHotInput, double[] targetOutputOneHot)
    {
        FeedForward(concatenatedOneHotInput);

        Layer outputLayer = _mlpLayers.Last();
        for (int i = 0; i < outputLayer.Neurons.Count; i++)
        {
            Neuron neuron = outputLayer.Neurons[i];
            neuron.Delta = neuron.Output - targetOutputOneHot[i];
        }

        for (int l = _mlpLayers.Count - 2; l >= 0; l--)
        {
            Layer currentHiddenLayer = _mlpLayers[l];
            Layer nextLayer = _mlpLayers[l + 1];

            for (int i = 0; i < currentHiddenLayer.Neurons.Count; i++)
            {
                Neuron neuron = currentHiddenLayer.Neurons[i];
                double errorSum = 0;
                for (int j = 0; j < nextLayer.Neurons.Count; j++)
                {
                    errorSum += nextLayer.Neurons[j].Weights[i] * nextLayer.Neurons[j].Delta;
                }
                neuron.Delta = errorSum * currentHiddenLayer.ActivationDerivative(currentHiddenLayer.Neurons[i].Output);

                double[] dropoutMask = currentHiddenLayer.GetLastDropoutMask();
                if (dropoutMask != null)
                {
                    neuron.Delta *= dropoutMask[i];
                }
            }
        }

        for (int l = 0; l < _mlpLayers.Count; l++)
        {
            Layer currentLayer = _mlpLayers[l];
            double[] actualInputsToCurrentLayer = _mlpLayerInputsCache[l];

            foreach (Neuron neuron in currentLayer.Neurons)
            {
                for (int i = 0; i < neuron.Weights.Length; i++)
                {
                    if (i < actualInputsToCurrentLayer.Length)
                    {
                        neuron.Weights[i] -= _learningRate * neuron.Delta * actualInputsToCurrentLayer[i];
                    }
                }
                neuron.Bias -= _learningRate * neuron.Delta;
            }
        }
    }

    public void Train(List<Tuple<double[], double[]>> trainingData, int epochs)
    {
        if (trainingData == null || trainingData.Count == 0)
        {
            Console.WriteLine("Uyarı: Eğitim verisi boş. Eğitim atlanıyor.");
            return;
        }
        Console.WriteLine($"Eğitim başlıyor: {epochs} epoch, {trainingData.Count} örnek.");

        SetTrainingMode(true);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            var shuffledData = trainingData.OrderBy(x => _random.Next()).ToList();

            foreach (var sample in shuffledData)
            {
                double[] oneHotInputs = sample.Item1;
                double[] targets = sample.Item2;
                double[] outputs = FeedForward(oneHotInputs);

                for (int i = 0; i < targets.Length; ++i)
                {
                    if (targets[i] == 1.0 && outputs[i] > 0)
                    {
                        totalError -= Math.Log(outputs[i] + 1e-9);
                    }
                }
                Backpropagate(oneHotInputs, targets);
            }
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Ortalama Kayıp: {totalError / trainingData.Count:F6}");
        }
        SetTrainingMode(false);
        Console.WriteLine("Eğitim tamamlandı.");
    }

    public void SaveModel(string filePath)
    {
        var modelData = new NeuralNetworkModelData
        {
            InputWindowSize = this._inputWindowSize,
            VocabularySize = this._vocabularySize,
            HiddenLayerSizes = this._hiddenLayerSizes,
            OutputSize = this._outputVocabularySize,
            LearningRate = this._learningRate,
            LayersData = new List<LayerModelData>()
        };

        foreach (var layer in this._mlpLayers)
        {
            var layerDto = new LayerModelData
            {
                ActivationFunctionName = layer.ActivationFunctionName,
                IsOutputLayerWithSoftmax = layer.IsOutputLayerWithSoftmax,
                DropoutRate = layer.DropoutRate,
                Neurons = layer.Neurons.Select(n => new NeuronModelData { Weights = n.Weights, Bias = n.Bias }).ToList()
            };
            modelData.LayersData.Add(layerDto);
        }

        string json = JsonConvert.SerializeObject(modelData, Formatting.Indented);
        File.WriteAllText(filePath, json);
        Console.WriteLine($"Model kaydedildi: {filePath}");
    }

    public static NeuralNetwork LoadModel(string filePath, Random randomSeedForNewOps = null)
    {
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Model dosyası bulunamadı: {filePath}");
            return null;
        }
        string json = File.ReadAllText(filePath);
        var modelData = JsonConvert.DeserializeObject<NeuralNetworkModelData>(json);

        Random currentRandom = randomSeedForNewOps ?? new Random();

        NeuralNetwork loadedNetwork = new NeuralNetwork(
            modelData.InputWindowSize,
            modelData.VocabularySize,
            modelData.HiddenLayerSizes,
            modelData.LearningRate,
            currentRandom
        );

        if (loadedNetwork._mlpLayers.Count != modelData.LayersData.Count)
        {
            Console.WriteLine("Hata: Yüklenen modeldeki MLP katman sayısı oluşturulan ağ ile uyuşmuyor.");
            return null;
        }

        for (int i = 0; i < modelData.LayersData.Count; i++)
        {
            var layerData = modelData.LayersData[i];
            var targetLayer = loadedNetwork._mlpLayers[i];

            targetLayer.ActivationFunctionName = layerData.ActivationFunctionName;
            targetLayer.AssignActivationFunctions();
            targetLayer.IsOutputLayerWithSoftmax = layerData.IsOutputLayerWithSoftmax;
            targetLayer.DropoutRate = layerData.DropoutRate;

            if (targetLayer.Neurons.Count != layerData.Neurons.Count)
            {
                Console.WriteLine($"Hata: MLP Katman {i} için nöron sayısı uyuşmuyor."); return null;
            }
            for (int j = 0; j < layerData.Neurons.Count; j++)
            {
                targetLayer.Neurons[j].Weights = layerData.Neurons[j].Weights;
                targetLayer.Neurons[j].Bias = layerData.Neurons[j].Bias;
            }
        }
        loadedNetwork.SetTrainingMode(false);
        Console.WriteLine($"Model yüklendi: {filePath}");
        return loadedNetwork;
    }
}

public class Program
{
    public static double[] GetInputVectorForWindow(string[] windowWords, Vocabulary vocab, int expectedWindowSize)
    {
        List<double> concatenatedOneHot = new List<double>();
        for (int i = 0; i < expectedWindowSize; i++)
        {
            string wordInWindow = (i < windowWords.Length) ? windowWords[i] : Vocabulary.PadToken;
            if (i >= windowWords.Length && windowWords.Length < expectedWindowSize)
            {
                wordInWindow = Vocabulary.PadToken;
            }
            concatenatedOneHot.AddRange(vocab.GetOneHotVector(wordInWindow));
        }
        return concatenatedOneHot.ToArray();
    }

    public static List<Tuple<double[], double[]>> PrepareTrainingData(
        IEnumerable<string[]> corpus, Vocabulary vocab, int windowSize)
    {
        var trainingPairs = new List<Tuple<double[], double[]>>();
        if (windowSize < 1) { Console.WriteLine("Uyarı: Pencere boyutu en az 1 olmalıdır."); return trainingPairs; }

        foreach (var sequence in corpus)
        {
            if (sequence == null || sequence.Length == 0) continue;
            for (int j = 0; j < sequence.Length; j++)
            {
                string targetWord = sequence[j];
                string[] currentWindowWords = new string[windowSize];
                for (int k = 0; k < windowSize; k++)
                {
                    int sourceIndex = j - windowSize + k;
                    currentWindowWords[k] = (sourceIndex < 0) ? Vocabulary.PadToken : sequence[sourceIndex];
                }

                double[] inputOneHotVector = GetInputVectorForWindow(currentWindowWords, vocab, windowSize);
                double[] targetVector = vocab.GetOneHotVector(targetWord);

                trainingPairs.Add(Tuple.Create(inputOneHotVector, targetVector));
            }
        }
        return trainingPairs;
    }

    public static void Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("Yapay Sinir Ağı (One-Hot Girişli) ile Kelime Sırası Öğrenme...");

        string vocabPath = "vocabulary_onehot.json";
        string modelPath = "nn_model_onehot.json";

        int windowSize = 2;
        double learningRate = 0.02;
        int epochs = 1000;
        int[] hiddenLayerSizes = new int[] { 32 };
        Random random = new Random(42);

        Vocabulary vocabulary = new Vocabulary();
        NeuralNetwork nn;

        Console.WriteLine("Kaydedilmiş model ve kelime dağarcığı yüklensin mi? (e/h)");
        string choice = Console.ReadLine().ToLower();

        if (choice == "e" && File.Exists(modelPath) && File.Exists(vocabPath))
        {
            Console.WriteLine("Kaydedilmiş model ve kelime dağarcığı yükleniyor...");
            vocabulary = Vocabulary.Load(vocabPath);
            nn = NeuralNetwork.LoadModel(modelPath, random);
            if (vocabulary == null || nn == null)
            {
                Console.WriteLine("Model veya kelime dağarcığı yüklenemedi. Yeni model oluşturulacak.");
                vocabulary = new Vocabulary();
                nn = null;
            }
        }
        else
        {
            Console.WriteLine("Yeni model oluşturuluyor ve eğitilecek...");
            nn = null;
        }

        if (nn == null)
        {
            var corpus = new List<string[]>
            {
                new string[] { "kırmızı", "ışık", "yanınca", "dur" },
                new string[] { "yeşil", "ışık", "yanınca", "geç" },
                new string[] { "sarı", "ışık", "yanınca", "hazırlan" },
                new string[] { "bir", "iki", "üç", "dört", "beş" },
                new string[] { "elma", "armut", "portakal", "mandalina" },
                new string[] { "pazartesi", "salı", "çarşamba", "perşembe", "cuma" },
                new string[] { "kedi", "köpek", "kuş", "balık" },
                new string[] { "anne", "baba", "çocuk", "aile" },
                new string[] { "sabah", "öğle", "akşam", "gece" },
                new string[] { "git", "gel", "otur", "kalk" },
                new string[] { "kırmızı", "ışık", "dur" },
                new string[] { "yeşil", "ışık", "geç" },
            };

            vocabulary = new Vocabulary();
            vocabulary.Build(corpus);
            Console.WriteLine($"Kelime Dağarcığı Boyutu: {vocabulary.Size}");

            var trainingData = PrepareTrainingData(corpus, vocabulary, windowSize);
            if (trainingData.Count == 0 && windowSize > 0) { Console.WriteLine("Eğitim verisi oluşturulamadı."); return; }
            Console.WriteLine($"Eğitim için {trainingData.Count} örnek çifti oluşturuldu.");

            nn = new NeuralNetwork(windowSize, vocabulary.Size,
                                     hiddenLayerSizes, learningRate, random);
            Console.WriteLine($"Sinir Ağı Oluşturuldu: Pencere={windowSize}, Vocab={vocabulary.Size}, Gizli={string.Join(",", hiddenLayerSizes)}");

            nn.Train(trainingData, epochs);

            vocabulary.Save(vocabPath);
            nn.SaveModel(modelPath);
        }

        nn.SetTrainingMode(false);

        Console.WriteLine("\n--- Tahmin Testi ---");
        while (true)
        {
            Console.WriteLine("\nTahmin için bir başlangıç dizisi girin (kelimeleri boşlukla ayırın, çıkmak için 'q' yazın):");
            string userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput)) continue;
            if (userInput.ToLower() == "q") break;

            string[] testSequenceWords = userInput.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (testSequenceWords.Length == 0) continue;

            string[] currentTestWindowWords = new string[windowSize];
            for (int k = 0; k < windowSize; k++)
            {
                int sourceIndex = testSequenceWords.Length - windowSize + k;
                currentTestWindowWords[k] = (sourceIndex < 0) ? Vocabulary.PadToken : testSequenceWords[sourceIndex];
            }

            double[] testInputVector = GetInputVectorForWindow(currentTestWindowWords, vocabulary, windowSize);
            Console.WriteLine($"Giriş penceresi (kelimeler): [{string.Join(", ", currentTestWindowWords.Select(w => $"'{w}'"))}]");

            double[] predictionOutput = nn.FeedForward(testInputVector);

            var sortedPredictions = predictionOutput
                .Select((probability, index) => new { Index = index, Probability = probability })
                .OrderByDescending(x => x.Probability)
                .Take(3);

            Console.WriteLine("==> Modelin Tahminleri:");
            int count = 0;
            foreach (var pred in sortedPredictions)
            {
                count++;
                string predictedWord = vocabulary.GetWord(pred.Index);
                Console.WriteLine($"{count}. {predictedWord} (Olasılık: {pred.Probability:P2})");
            }
            if (count == 0) Console.WriteLine("Tahmin yapılamadı.");
        }

        Console.WriteLine("\nProgram tamamlandı.");
    }
}
