import random
import numpy as np

class NeuralNetwork:
    def __init__(self, _input_size, _output_size, _list_of_hidden_sizes, _learning_rate):
        print("\n[OK] Starting Neural Network Inizialization")
        # Neural Network Parameters Storing:
        self.input_size = _input_size                               # Val which contains the size of Input Layer
        self.output_size = _output_size                             # Val which contains the size of Output Layer
        self.list_of_hidden_sizes = _list_of_hidden_sizes           # List which contains the size of each Hidden Layer
        self.hidden_layers_number = len(self.list_of_hidden_sizes)  # Val which contains the number of Hidden Layers
        self.theta_matrix_number = self.hidden_layers_number + 1    # Val which contains the number of Weights Matrixs
        self.learning_rate = _learning_rate                         # Val which contains the NN's Learning rate

        # Input/Output Layers Creation:
        self.input = np.zeros((self.input_size))
        self.output = np.zeros((self.output_size))
        # NB: We don't need the derivative terms for Input Layer and Output Layer

        # Hidden Layers Creation:
        self.hidden_layers         = [[] for x in range(self.hidden_layers_number)] # List of Empty List
        self.hidden_layers_derived = [[] for x in range(self.hidden_layers_number)] # List of Empty List
        for i in range(self.hidden_layers_number):
            self.hidden_layers[i]         = np.zeros((self.list_of_hidden_sizes[i])) # Each Hidden Layer to Zero
            self.hidden_layers_derived[i] = np.ones((self.list_of_hidden_sizes[i]))  # Each Hidden Layer Derivative to One

        # Weights Matrix Creation:
        self.theta_matrixs = [[] for x in range(self.theta_matrix_number)]
        for i in range(self.theta_matrix_number):
            if i == 0:
                self.theta_matrixs[i] = np.random.rand(self.input_size, self.hidden_layers[i].size) # Random Weights from 0 to 1
                # self.theta_matrixs[i] = np.ones((self.input_size, self.hidden_layers[i].size))
            elif i == self.theta_matrix_number-1:
                self.theta_matrixs[i] = np.random.rand(self.hidden_layers[i-1].size, self.output_size) # Random Weights from 0 to 1
                # self.theta_matrixs[i] = np.ones((self.hidden_layers[i-1].size, self.output_size))
            else:
                self.theta_matrixs[i] = np.random.rand(self.hidden_layers[i-1].size, self.hidden_layers[i].size) # Random Weights from 0 to 1
                # self.theta_matrixs[i] = np.ones((self.hidden_layers[i-1].size, self.hidden_layers[i].size))

        print("[OK] Neural Network Inizialization Completed\n")
    
    # Possible Activation Functions/Derivatives:
    def identity_function(self, array):
        return array
    def identity_derivative(self, array):
        return 1
    def sigmoid_function(self, array):
        return 1/(1 + np.exp(-array))
    def sigmoid_derivative(self, array):
        return self.sigmoid_function(array) * (1 - self.sigmoid_function(array)) 
    
    # Neural Network Activation Function/Derivative:
    def activation_function(self, array):
        return self.identity_function(array)
    def activation_derivative(self, array):
        return self.identity_derivative(array)

    def forward_propagation(self, input_feature):
        self.input = np.array(input_feature)

        # NB: np.dot() stands for Matrix Prod
        for i in range(self.hidden_layers_number+1):
            # First Hidden Layer:
            if i == 0:
                self.hidden_layers[i] = np.dot(self.input, self.theta_matrixs[i])
                self.hidden_layers[i] = self.activation_function(self.hidden_layers[i])
                self.hidden_layers_derived[i] = self.activation_derivative(self.hidden_layers[i])
            # Output Layer:
            elif i == self.hidden_layers_number:
                self.output = np.dot(self.hidden_layers[i-1], self.theta_matrixs[i])
            # Others Hidden Layers:
            else:
                self.hidden_layers[i] = np.dot(self.hidden_layers[i-1], self.theta_matrixs[i])
                self.hidden_layers[i] = self.activation_function(self.hidden_layers[i])
                self.hidden_layers_derived[i] = self.activation_derivative(self.hidden_layers[i])

    def back_propagation(self, output_feature):
        output_error = self.output - np.array(output_feature)
        # Array to Matrix, because we need to do Matrix Prod (which is not allowed if output_error is an Array)
        output_error = np.array([output_error])

        # Calculating the error for each Layer:
        hidden_layers_error = [[] for x in range(self.hidden_layers_number)]
        for i in range(self.hidden_layers_number-1, -1, -1):
            if i == self.hidden_layers_number-1:
                hidden_layers_error[i] = np.dot(output_error, self.theta_matrixs[i+1].T) * self.hidden_layers_derived[i]
            else:
                hidden_layers_error[i] = np.dot(hidden_layers_error[i+1], self.theta_matrixs[i+1].T) * self.hidden_layers_derived[i]

        # Weights adjustment (Gradient Descent):
        for i in range(self.theta_matrix_number-1, -1, -1):
            # Last Weights Matrix:
            if i == self.theta_matrix_number-1:
                self.theta_matrixs[i] = self.theta_matrixs[i] - self.learning_rate * (np.dot(np.array([self.hidden_layers[i-1]]).T, output_error))
            # First Weights Matrix:
            elif i == 0:
                self.theta_matrixs[i] = self.theta_matrixs[i] - self.learning_rate * (np.dot(np.array([self.input]).T, hidden_layers_error[i]))
            # Other Weights Matrix:
            else:
                self.theta_matrixs[i] = self.theta_matrixs[i] - self.learning_rate * (np.dot(np.array([self.hidden_layers[i-1]]).T, hidden_layers_error[i]))
    
    def print_NN(self, cmd):
        if(cmd == "arch"):
            print("-----------------------------------")
            print("Neural Network Architecture:")
            print("-----------------------------------")
            print("Input Layer   :", self.input_size)
            print("Hidden Layers :", self.list_of_hidden_sizes)
            print("Output Layer  :", self.output_size)
            print("Learning Rate :", self.learning_rate)

        elif(cmd == "layers"):
            print("-----------------------------------")
            print("Neural Network Layers:")
            print("-----------------------------------")
            print("Input Layer              :", self.input)
            for i in range(self.hidden_layers_number):
                print("Hidden Layer ", i, "         :", self.hidden_layers[i])
                print("Hidden Layer Derived ", i, " :", self.hidden_layers_derived[i])
            print("Output Layer             :", self.output)
        elif(cmd == "theta"):
            print("-----------------------------------")
            print("Neural Networks Weights:")
            print("-----------------------------------")
            for i in range(self.theta_matrix_number):
                print("Theta Matrix ", i, " :")
                print(self.theta_matrixs[i])
        elif(cmd == "hypo"):
            print("Input Layer  :", self.input)
            print("Output Layer :", self.output)
        else:
            print("ERROR NOT A VALID CMD")

if __name__ == "__main__":
    # NeuralNetwork(size_of_input_layer, size_of_output_layer, [list_with_sizes_of_each_hidden_layer], learning_rate)
    nn = NeuralNetwork(2, 1, [5,5,5,5], 0.00001)
    nn.print_NN("layers")
    # nn.print_NN("arch")
    # nn.print_NN("theta")    

    print("\nNot Trained NN Hypotesis:")
    print("-------------------------")
    nn.forward_propagation([2,2])
    nn.print_NN("hypo")

    number_of_iterations = 10000
    print("\n[OK] Training Started")
    for i in range(number_of_iterations):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        nn.forward_propagation([a,b])
        nn.back_propagation([a+b])
    print("[OK] Training Completed\n")

    print("Trained NN Hypotesis: ")
    print("-------------------------")
    nn.forward_propagation([4,4])
    nn.print_NN("hypo")
    nn.forward_propagation([9,9])
    nn.print_NN("hypo")
    nn.forward_propagation([7,7])
    nn.print_NN("hypo")