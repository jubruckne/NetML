namespace NetML.ML;

public class Trainer {
    public Network network { get; }
    public float learning_rate { get; }

    public Trainer(Network network, float learning_rate) {
        this.network = network;
        this.learning_rate = learning_rate;
    }

    public void train(Dataset dataset) {
        //
    }
}