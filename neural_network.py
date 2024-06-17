# Importing modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Custom Neural Network Class
class Custom_Neural_Network:
    """
        This class creates a custom neural network.
        The neural network is a Sequential model.
        The hidden layers are Dense layers that use the relu activation function.
        The output layer is a single node, Dense layer using the sigmoid activation function
        
        @PARAMS:
        - num_layers: integer value of number of hidden layers
        - number_input_features: integer value of the number of features within the featureset
        
        @METHODS:
        - __init__: Class constructor
        - on_init: Method to handle output and set variables
        - set_layers: Method to set the number of nodes for each hidden layer 
        - print_layers: Method to print layers variable values
        - set_neural_network_layers: Method to set layers for neural network
        - get_neural_network: Method to return neural network
        
    """
    
    # Class constuctor
    def __init__(self, number_layers, number_input_features):
        
        """
            Initializes variables and calls on_init method
        """
        
        # Setting Class Variables
        self.number_layers = number_layers
        self.number_input_features = number_input_features
        self.current_layer = number_input_features
        self.number_output_neurons = 1
        self.layers = []
        self.relu = 'relu'
        self.sigmoid = 'sigmoid'
        self.nn = Sequential()
        self.on_init()
        
        # Method to handle output and set variable values
    def on_init(self):
        
        """
            Handles output and sets variable values
        """
        # Printing the number of input features
        print(self.number_input_features)
        # Setting layers
        self.set_layers()
        # Printing layers
        self.print_layers()
        # Setting neural network layers
        self.set_neural_network_layers()
        # Displaying neural network summary
        display(self.nn.summary())
        
        # Method to set the number of nodes for each hidden layer
    def set_layers(self):
        
        """
            Sets the number of nodes for each hidden layer
        """
        
        # Looping over number of layers
        for i in range(self.number_layers):
            # Setting current layer values
            self.current_layer = (self.current_layer + 1) // 2
            # Appending current layer to layers list
            self.layers.append(self.current_layer)
                
        # Method to print layer values        
    def print_layers(self):
        
        """
            Number of nodes for each hidden layer
        """
        # Looping over layer values
        for i in self.layers:
            # Printing layer value
            print(i)
            
        # Method to set the layers for the neural network    
    def set_neural_network_layers(self):
        
        """
            Adds the layers of the neural network
        """
        # Creating the first layer
        self.nn.add(Dense(self.layers[0], input_dim=self.number_input_features, activation=self.relu))
        
        # Looping over a range of the layers
        for i in range(1,len(self.layers)):
            # Adding each hidden layer
            self.nn.add(Dense(self.layers[i], activation=self.relu))
        # Adding the final output layer
        self.nn.add(Dense(self.number_output_neurons, activation=self.sigmoid))
        
        # Method to return neural network
    def get_neural_network(self):
        
        """
            Returns the neural network itself
        """
        # Returning the neural network
        return self.nn