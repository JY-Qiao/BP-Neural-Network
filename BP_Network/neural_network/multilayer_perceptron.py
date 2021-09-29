import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient



class MultilayerPerceptron:
    def __init__(self,data,labels,layers,normalize_data =False):
        data_processed = prepare_for_training(data,normalize_data = normalize_data)[0]
        self.data= data_processed
        self.labels= labels
        self.layers= layers
        self.normalize_data= normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(layers)
        
    def predict(self,data):
        data_processed = prepare_for_training(data,normalize_data = self.normalize_data)[0]

        predictions = MultilayerPerceptron.feedforward_propagation(data_processed,self.thetas,self.layers)
        
        return predictions
        
        
        
    def train(self,max_iterations=1000,alpha=0.1):
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)
        
        (optimized_theta,cost_history) = MultilayerPerceptron.gradient_descent(self.data,self.labels,unrolled_theta,self.layers,max_iterations,alpha)
        
        
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta,self.layers)

        MultilayerPerceptron.write_file((self.thetas[0]).T)
        return self.thetas,cost_history
         
    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            """
                            Execute twice to get two sets of parameter matrices
            """
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]

            thetas[layer_index] = np.random.rand(out_count,in_count+1)*0.05 # Random initialization
        return thetas
    
    @staticmethod
    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            unrolled_theta = np.hstack((unrolled_theta,thetas[theta_layer_index].flatten()))
        return unrolled_theta
    
    @staticmethod
    def gradient_descent(data,labels,unrolled_theta,layers,max_iterations,alpha):
        
        optimized_theta = unrolled_theta
        cost_history = []
        
        for _ in range(max_iterations):

            cost = MultilayerPerceptron.cost_function(data,labels,MultilayerPerceptron.thetas_roll(optimized_theta,layers),layers)
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data,labels,optimized_theta,layers)
            optimized_theta = optimized_theta - alpha * theta_gradient
        return optimized_theta,cost_history
            
            
    @staticmethod 
    def gradient_step(data,labels,optimized_theta,layers):
        theta = MultilayerPerceptron.thetas_roll(optimized_theta,layers)
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data,labels,theta,layers)
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients
    
    @staticmethod 
    def back_propagation(data,labels,thetas,layers):
        num_layers = len(layers)
        (num_examples,num_features) = data.shape

        
        deltas = {}
        # Initialization
        for layer_index in range(num_layers -1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            deltas[layer_index] = np.zeros((out_count,in_count+1))
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index,:].reshape((num_features,1))
            layers_activations[0] = layers_activation
            # Calculate by layer
            for layer_index in range(num_layers - 2):
                layer_theta = thetas[layer_index]
                layer_input = np.dot(layer_theta,layers_activation)
                layers_activation = np.vstack((np.array([[1]]),sigmoid(layer_input)))
                layers_inputs[layer_index + 1] = layer_input
                layers_activations[layer_index + 1] = layers_activation
            # Calculation of the last layer (without sigmoid)
            layer_index = num_layers - 2
            layer_theta = thetas[layer_index]
            layer_input = np.dot(layer_theta,layers_activation)
            layers_activation = np.vstack((np.array([[1]]),layer_input))
            layers_inputs[layer_index + 1] = layer_input
            layers_activations[layer_index + 1] = layers_activation
            output_layer_activation = layers_activation[1:,:]

            
            delta = {}
            # calculate the diff between the prediction and the precision
            bitwise_label = labels[example_index]
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # calculate the diff between the penultimate layer value and the precision
            layer_index = num_layers - 2
            layer_theta = thetas[layer_index]
            next_delta = delta[layer_index + 1]

            delta[layer_index] = np.dot(layer_theta.T, next_delta)
            delta[layer_index] = delta[layer_index][1:,:]

            # Traversal loop L L-1 L-2 ...2
            for layer_index in range(num_layers - 3,0,-1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)),layer_input))

                delta[layer_index] = np.dot(layer_theta.T,next_delta)*sigmoid_gradient(layer_input)
                # Filter out bias parameters
                delta[layer_index] = delta[layer_index][1:,:]
            for layer_index in range(num_layers-1):
                layer_delta = np.dot(delta[layer_index+1],layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta
                
        for layer_index in range(num_layers -1):
               
            deltas[layer_index] = deltas[layer_index] * (1/num_examples)
            
        return deltas
            
    @staticmethod        
    def cost_function(data,labels,thetas,layers):
        cost = 0

        predictions = MultilayerPerceptron.feedforward_propagation(data,thetas,layers)
        predictions = predictions.flatten()
        length = len(predictions)

        labels = labels.flatten()
        distances = labels - predictions
    
        for distance in distances:
            cost = cost + distance ** 2
        cost = (1/length) * cost
        return cost
                
    @staticmethod        
    def feedforward_propagation(data,thetas,layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data
        
        # Calculate by layer
        for layer_index in range(num_layers - 2):
            theta = thetas[layer_index]
            out_layer_activation = np.dot(in_layer_activation,theta.T)
            # Add bias parameters
            out_layer_activation = np.hstack((np.ones((num_examples,1)),sigmoid(out_layer_activation)))
            in_layer_activation = out_layer_activation

        # Calculate the last year
        theta = thetas[num_layers - 2]
        out_layer_activation = np.dot(in_layer_activation, theta.T)
        in_layer_activation = out_layer_activation
        # Return results
        return in_layer_activation[:,:]
                   
    @staticmethod       
    def thetas_roll(unrolled_thetas,layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height,thetas_width))
            unrolled_shift = unrolled_shift+thetas_volume
        
        return thetas

    @staticmethod
    def write_file(thetas):
        f = open('thetas.txt', 'w')
        m, n = thetas.shape
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(thetas[i, j]))
            f.write('\t'.join(tmp) + '\n')
        f.close()