
import numpy as np
import copy

import torch
import torch.nn as nn

import torch.optim as optim

from scipy import stats

from scipy.special import softmax

from media import Media




class DeepNeuralNetwork(nn.Module):

    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()

        self.input_layer_len = 100
        self.output_layer_len = 1

        self.hidden_layer_1_len = 500
        self.hidden_layer_2_len = 500


        self.hidden_1 = nn.Linear(self.input_layer_len, self.hidden_layer_1_len, bias=True)

        self.hidden_2 = nn.Linear(self.hidden_layer_1_len, self.hidden_layer_2_len, bias=True)

        self.output = nn.Linear(self.hidden_layer_2_len, self.output_layer_len, bias=True)

        self.activation_on_hidden_1 = nn.ReLU() # nn.Tanh()

        self.activation_on_hidden_2 = nn.ReLU() # nn.Tanh()

        self.activation_on_output = nn.ReLU() # nn.Tanh()  # nn.Softmax()

        self.last_layer_before_activation = []


    def forward(self, x):

        x = self.hidden_1(x)
        x = self.activation_on_hidden_1(x)
        x = self.hidden_2(x)
        x = self.activation_on_hidden_2(x)
        x = self.output(x)

        self.last_layer_before_activation = copy.deepcopy(x.detach().numpy())

        x = self.activation_on_output(x)

        return x




class Agent():

    # Эти существа умеют находить себе пару. Если пара окажется успешной в том смысле,
    # что медиа контенты партнеров в слиянии произведут успешный медиа контент, то оба напарника-агента живут дольше
    # и получают больше ресурсов. Таким образом успешные агенты с большей вероятностью оставят потомство.
    # Потомство производится путем скрещивания двух агентов. Причем не обязательно, чтобы агент скрещивал свой
    # геном с тем же напарником, с которым скрещивает медиа контент. Однако для скрещивания генома агент
    # предпочтет агента с более высоким "статусом" ("статус" это один из ресурсов). Геном представляет из себя
    # набор микропараметров агента, включая параметры нейронной сети агента. Нейронная сеть агента принимает
    # в некотором виде его медиа контент, медиа контент потенциального напарника и пытается предсказать оценку
    # пользователья результирующего (слитого) медиаконтента. Таким образом, агент может отранжировать некоторый
    # набор других агентов и выбрать из них того, который позволит им произвести наиболее успешный медиа контент.
    # При выборе агента-напарника агент (пока не знаю каким образом) учитывает не только свою сортировку
    # потенциальных напарников, но и их предпочтения относительно себя.

    ## Правила
    # После увеличения self.media_content.score self.lifetime увеличивается на x < y. Также увеличивается self.status.
    # После каждого скрещивания self.lifetime сокращается на y.
    # После уменьшения self.media_content.score уменьшается self.status на z.


    def __init__(self):

        self.media_content = None # Media

        self.couple = None
        # self.couple_ID = None
        # self.couple_name = None

        self.genome = None

        self.lifetime = None
        self.status = None

        ## Agent's Deep Neural Network
        self.DNN = DeepNeuralNetwork()
        self.DNN_shapes = []
        orig_DNN = copy.deepcopy(self.DNN)
        for param in orig_DNN.parameters():
            p = param.data.cpu().numpy()
            self.DNN_shapes.append(p.shape)

        ## Hyperparameters for fitting
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.DNN.parameters(), lr=0.02)
        self.sigma = 0.01


    def get_hash(self) -> int:

        # TODO

        pass


    def get_media_content(self) -> Media:
        return self.media_content


    def updateParams(self, flat_param: np.array):
        idx = 0
        i = 0

        for param in self.DNN.parameters():
            delta = np.product(self.DNN_shapes[i])
            block = flat_param[idx:idx + delta]
            block = np.reshape(block, self.DNN_shapes[i])
            i += 1
            idx += delta
            block_data = torch.from_numpy(block).float()

            param.data = block_data


    def fit(self, inputs, correct_outputs):

        correct_outputs = torch.from_numpy(correct_outputs).float()
        inputs = torch.from_numpy(inputs).float()

        self.optimizer.zero_grad()

        guesses = self.DNN(inputs)
        loss = self.criterion(guesses, correct_outputs)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def format_DNN_input(self, media_content_1, media_content_2):

        #TODO

        pass


    def predict_success_of_symbiosis(self, potential_couple_agent):

        assert potential_couple_agent.get_hash() != self.get_hash(), "Agent can not create symbiosis with its self!"

        input = self.format_DNN_input(self.get_media_content(), potential_couple_agent.get_media_content())

        output = self.DNN(torch.from_numpy(input).float()).detach().numpy()

        return output


    def find_couple(self, population):

        # чем больше статус, тем больше вероятность, что агента выберут в качестве пары

        statuses = population.get_status_of_each_agent()
        probabilities = softmax(np.array(statuses))
        agents_indexes = np.arange(len(statuses))

        custm = stats.rv_discrete(name='custm', values=(agents_indexes, probabilities))

        couple_index = custm.rvs(size=1)[0]

        couple = population[couple_index]

        return couple


    def crossing(self):

        #TODO read about neuro evolutionary

        pass



class PopulationOfAgents:

    def __init__(self):

        self.array_of_agents_in_population = []


    def get_population_size(self):

        return len(self.array_of_agents_in_population)


    def get_status_of_each_agent(self) -> list:

        return [agent.status for agent in self.array_of_agents_in_population]


    def add_agent(self, agent: Agent):

        self.array_of_agents_in_population.append(agent)


    def remove_agent(self, agent: Agent):

        assert agent in self.array_of_agents_in_population, "The agent dose not exist!"

        self.array_of_agents_in_population.remove(agent)
