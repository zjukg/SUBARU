import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple

from transformers import LlamaForCausalLM



class KnowledgePrompting(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        kge_model: str = "data/transe.pt",
        pretrain_emb_path = None,
        adapter_type = "mlp"
    ) -> None:
        super(KnowledgePrompting, self).__init__()
        self.llama_model = model
        for param in self.llama_model.parameters():
            param.requires_grad = False
        pretrain_embeddings = torch.load(open(kge_model, "rb"))
        if pretrain_emb_path is None:
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=pretrain_embeddings,
                dim_llm=4096,
                adapter_type=adapter_type
            )
        else:
            print("Adapter Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)
        print(self.embeddings)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llama_model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PretrainKGEmbedding(nn.Module):
    def __init__(
        self,
        pretrain_ent_embs,
        dim_llm,
        num_prefix = 1,
        adapter_type = "mlp"
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.pretrain_dim = self.embeddings.weight.shape[1]
        # Froze the pretrain embeddings
        self.embeddings.requires_grad_(False)
        self.adapter_type = adapter_type
        if adapter_type == "fc":
            self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)
        elif adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(self.pretrain_dim, 3 * self.emb_dim),
                nn.ReLU(),
                nn.Linear(3 * self.emb_dim, self.emb_dim)
            )
        elif adapter_type == "moe":
            self.adapter = MoEAdaptorLayer(layers=[self.pretrain_dim, self.emb_dim])
        elif adapter_type == "qformer":
            self.adapter = QFormer(self.pretrain_dim, self.emb_dim)
        elif "mlp_" in adapter_type:
            # The scalability
            num_layers = int(adapter_type.split('_')[-1])
            self.adapter = nn.Sequential(
                nn.Linear(self.pretrain_dim, 3 * self.emb_dim),
                nn.ReLU(),
            )
            for _ in range(num_layers - 2):
                self.adapter.append(nn.Linear(3 * self.emb_dim, 3 * self.emb_dim))
                self.adapter.append(nn.ReLU())
            self.adapter.append(nn.Linear(3 * self.emb_dim, self.emb_dim))
        elif "res_" in adapter_type:
            pass
        else:
            raise NotImplementedError
    

    def forward(self, triple_ids):
        # main training stage
        batch_size = triple_ids.shape[0]
        num_token = triple_ids.shape[1]
        ent = triple_ids.reshape(-1, num_token)
        with torch.no_grad():
            emb = self.embeddings(ent)
        prefix = self.adapter(emb).reshape(batch_size, -1, self.llm_dim)
        # print(prefix.shape)
        return prefix


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
    

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps=4, layers=[512, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class QFormer(nn.Module):
    def __init__(self, hidden_dim, query_dim, num_queries=4, num_heads=1, num_layers=1):
        super(QFormer, self).__init__()
        # 查询向量的嵌入
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))
        # Query Transformer
        self.query_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, query_dim)
        

    def forward(self, x):
        queries = self.query_embeddings.unsqueeze(0).repeat(x.size(0), 1, 1)
        query_output, _ = self.query_self_attention(queries, queries, queries)
        cross_attention_output, _ = self.cross_attention(query_output, x, x)
        output = self.feed_forward(cross_attention_output) + cross_attention_output
        output = self.fc_out(self.layer_norm(output))
        return output


if __name__ == "__main__":
    qformer = QFormer(hidden_dim=512, query_dim=4096)
    input = torch.randn((16, 1, 512))
    print(qformer(input).shape)
