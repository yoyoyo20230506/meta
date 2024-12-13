class STDPModule(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.01, beta=0.02):
        super(STDPModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.beta = beta
        self.synaptic_weights = nn.Parameter(torch.rand(output_dim, input_dim))

    def forward(self, x):
        spike_timings = torch.matmul(x, self.synaptic_weights.T)
        dw = self.alpha * spike_timings + self.beta * torch.rand_like(spike_timings)
        self.synaptic_weights.data += dw
        return self.fc(x)

if __name__ == "__main__":
    x = torch.randn(1, 10)
    stdp = STDPModule(10, 5)
    output = stdp(x)
    print("STDP output:", output)
