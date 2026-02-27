from chronosrep import ChronosRepModel

if __name__ == "__main__":
    model = ChronosRepModel()
    for _ in range(ChronosRepModel.T):
        model.step()
    print(f"Simulation complete: {ChronosRepModel.N} agents, {ChronosRepModel.T} steps, τ={ChronosRepModel.TAU}")
