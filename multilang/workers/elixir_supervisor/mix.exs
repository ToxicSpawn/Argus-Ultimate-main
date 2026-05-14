defmodule ArgusSupervisor.MixProject do
  use Mix.Project

  def project do
    [
      app: :argus_supervisor,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {ArgusSupervisor.Application, []}
    ]
  end

  defp deps do
    [
      # {:plug_cowboy, "~> 2.6"}  # For HTTP health endpoint
    ]
  end
end
