import torch
import torch.nn as nn
import torch.nn.functional as F



class Policy_step(nn.Module):
    def __init__(self, m, embedding_size, hidden_size):
        super(Policy_step, self).__init__()

        self.embedding_size = embedding_size
        self.m=m
        self.hidden_size = hidden_size
        self.transformer = nn.Transformer(d_model = m * embedding_size,
                                          num_encoder_layers=1,
                                          num_decoder_layers=1, nhead=1,
                                          dim_feedforward= m * self.hidden_size,
                                          norm_first=True
                                          )

        self.l2 = nn.Linear(2 * m * embedding_size, m * embedding_size)
        self.l3 = nn.Linear(m * embedding_size, m * embedding_size)


    def forward(self, prev_action, prev_state):


        prev_state = torch.relu(self.l2(prev_state))


        prev_action = prev_action.squeeze()
        batch = 20
        if prev_state.shape[0] % 128 == 0 :
            batch = 128
        prev_action = prev_action.squeeze()
        prev_action = prev_action.reshape(batch, prev_action.shape[0]//batch, prev_action.shape[1])
        prev_state = prev_state.squeeze()
        prev_state = prev_state.reshape(batch,prev_state.shape[0]//batch , prev_state.shape[1])

        output = self.transformer(prev_action, prev_state)
        output = output.reshape(batch * output.shape[1], output.shape[-1])

        ch = torch.relu(self.l3(output))

        output , ch = output.squeeze() , ch.squeeze()

        return output, ch

class Policy_mlp(nn.Module):
    def __init__(self, hidden_size, m, embedding_size):
        super(Policy_mlp, self).__init__()

        self.hidden_size = hidden_size
        self.m = m
        self.embedding_size = embedding_size
        self.mlp_l1 = nn.Linear(2 * m * self.hidden_size, m * self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(m * self.hidden_size, m * self.embedding_size, bias=True)

    def forward(self, state_query):
        # state_query = state_query.float()
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden))
        return output



class Agent(nn.Module):

    def __init__(self, params):
        super(Agent, self).__init__()
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']
        self.use_entity_embeddings = params['use_entity_embeddings']
        self.train_entity_embeddings = params['train_entity_embeddings']
        self.train_relation_embeddings = params['train_relation_embeddings']
        self.device = params['device']

        if self.use_entity_embeddings:
            if self.train_entity_embeddings:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
            else:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size).requires_grad_(
                    False)
            torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        else:
            if self.train_entity_embeddings:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
            else:
                self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size).requires_grad_(
                    False)
            torch.nn.init.constant_(self.entity_embedding.weight, 0.0)

        if self.train_relation_embeddings:
            self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size)
        else:
            self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size).requires_grad_(
                False)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = (torch.ones(self.batch_size) * params['relation_vocab']['DUMMY_START_RELATION']).long()
        # print(self.dummy_start_label.size())
        self.entity_embedding_size = self.embedding_size

        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        self.policy_step = Policy_step(m=self.m, embedding_size=self.embedding_size, hidden_size=self.hidden_size).to(self.device)
        self.policy_mlp = Policy_mlp(self.hidden_size, self.m, self.embedding_size).to(self.device)

        self.gate1_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)
        self.gate2_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)


    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)


    def action_encoder(self, next_relations, next_entities):
        # relation_embedding = self.relation_embedding[next_relations.cpu().numpy()]
        # entity_embedding = self.entity_embedding[next_entities.cpu().numpy()]
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)


        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
              range_arr, first_step_of_test, entity_cluster_shared_informs):


        prev_action_embedding = self.action_encoder(prev_relation, current_entities) # (original batch_size * num_rollout, 4*self.embedding_size)

        new_prev_state = list()

        prev_state = prev_state.squeeze()
        prev_state = torch.cat((prev_state,entity_cluster_shared_informs), dim=-1)

        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        # Get state vector
        prev_entity = self.entity_embedding(current_entities)
        if self.use_entity_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        query_embedding = self.relation_embedding(query_embedding)

        state_query_concat = torch.cat([state, query_embedding], dim=-1)

        output = self.policy_mlp(state_query_concat)
        output_expanded = torch.unsqueeze(output, dim=1)  # [original batch_size * num_rollout, 1, 2D], D=self.hidden_size
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)

        # Masking PAD actions

        comparison_tensor = torch.ones_like(next_relations).int() * self.rPAD  # matrix to compare
        mask = next_relations == comparison_tensor  # The mask
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = torch.where(mask, dummy_scores, prelim_scores)  # [original batch_size * num_rollout, max_num_actions]

        # 4 sample action
        action = torch.distributions.categorical.Categorical(logits=scores) # [original batch_size * num_rollout, 1]
        label_action = action.sample() # [original batch_size * num_rollout,]

        # loss
        loss = torch.nn.CrossEntropyLoss(reduce=False)(scores, label_action)

        # 6. Map back to true id
        chosen_relation = next_relations[list(torch.stack([range_arr, label_action]))]

        return loss, new_state, F.log_softmax(scores), label_action, chosen_relation


class EntityAgent(Agent):
    def __init__(self, params):
        Agent.__init__(self, params)



class ClusterAgent(nn.Module):
    def __init__(self, params):

        super(ClusterAgent, self).__init__()
        self.embedding_size = params['embedding_size']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.action_vocab_size = len(params['cluster_relation_vocab'])
        self.cluster_vocab_size = len(params['cluster_vocab'])
        self.device = params['device']
        self.use_cluster_embeddings = params['use_cluster_embeddings']
        self.hidden_size = params['hidden_size']
        self.rPAD = params['relation_vocab']['PAD']

        if self.use_cluster_embeddings:
            self.cluster_embedding = nn.Embedding(self.cluster_vocab_size, 2 * self.embedding_size)
            torch.nn.init.xavier_uniform_(self.cluster_embedding.weight)
        else:
            self.cluster_embedding = nn.Embedding(self.cluster_vocab_size, 2 * self.embedding_size)
            torch.nn.init.constant_(self.cluster_embedding.weight, 0.0)

        # print(self.dummy_start_label.size())
        self.cluster_embedding_size = self.embedding_size
        self.dummy_start_label = (torch.ones(self.batch_size) * params['cluster_relation_vocab']['DUMMY_START_RELATION']).long()

        if self.use_cluster_embeddings:
            self.m = 4
        else:
            self.m = 2

        self.policy_step = Policy_step(m=self.m, embedding_size=self.embedding_size, hidden_size=self.hidden_size).to(self.device)
        self.policy_mlp = Policy_mlp(self.hidden_size, self.m, self.embedding_size).to(self.device)

        self.gate1_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)
        self.gate2_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)

    def cluster_action_encoder(self, next_cluster, prev_cluster):
        # relation_embedding = self.relation_embedding[next_relations.cpu().numpy()]
        # entity_embedding = self.entity_embedding[next_entities.cpu().numpy()]

        next_cluster_emb = self.cluster_embedding(next_cluster)
        prev_cluster_emb = self.cluster_embedding(prev_cluster)

        if self.use_cluster_embeddings:
            action_embedding = torch.cat([next_cluster_emb, prev_cluster_emb], dim=-1)
        else:
            action_embedding = next_cluster_emb

        return action_embedding


    def cluster_step(self, prev_possible_clusters, next_clusters, prev_state, prev_cluster, end_cluster, current_clusters,
                     range_arr, first_step_of_test, entity_cluster_shared_informs):

        prev_action_embedding = self.cluster_action_encoder(prev_cluster, current_clusters)  # (original batch_size * num_rollout, 4*self.embedding_size)

        prev_state = torch.cat((prev_state,entity_cluster_shared_informs), dim=-1)

        # 1. one step of rnn
        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        # Get state vector
        prev_entity = self.cluster_embedding(current_clusters)
        if self.use_cluster_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output

        candidate_action_embeddings = self.cluster_action_encoder(prev_possible_clusters, next_clusters)

        query_embedding = self.cluster_embedding(end_cluster)


        state_query_concat = torch.cat([state, query_embedding], dim=-1)


        output = self.policy_mlp(state_query_concat)
        output_expanded = torch.unsqueeze(output, dim=1)  # [original batch_size * num_rollout, 1, 2D], D=self.hidden_size
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)

        # Masking PAD actions

        comparison_tensor = torch.ones_like(next_clusters).int() * self.rPAD  # matrix to compare
        mask = next_clusters == comparison_tensor  # The mask
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = torch.where(mask, dummy_scores, prelim_scores)  # [original batch_size * num_rollout, max_num_actions]

        # 4 sample action

        action = torch.distributions.categorical.Categorical(logits=scores)  # [original batch_size * num_rollout, max_num_actions]
        label_action = action.sample()  # [original batch_size * num_rollout,]

        # loss
        loss = torch.nn.CrossEntropyLoss(reduce=False)(scores, label_action)

        # 6. Map back to true id
        chosen_relation = next_clusters[list(torch.stack([range_arr, label_action]))]

        return loss, new_state, F.log_softmax(scores), label_action, chosen_relation, F.log_softmax(scores)
