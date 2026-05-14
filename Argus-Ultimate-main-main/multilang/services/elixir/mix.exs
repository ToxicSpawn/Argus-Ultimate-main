defmodule Argus.MixProject do
  use Mix.Project

  def project do
    [
      app: :argus,
      version: "1.0.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Argus Trading System - Elixir HTTP Service"
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto],
      mod: {Argus.Application, []}
    ]
  end

  defp deps do
    [
      {:plug_cowboy, "~> 2.7"},
      {:jason, "~> 1.4"}
    ]
  end
end
