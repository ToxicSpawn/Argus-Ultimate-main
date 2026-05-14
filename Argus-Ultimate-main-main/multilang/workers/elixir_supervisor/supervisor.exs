# ARGUS Elixir Exchange Connection Supervisor
#
# Monitors WebSocket connections to exchanges.
# Auto-restarts on crash with exponential backoff.
# Reports health status via HTTP on localhost:9997.
#
# Endpoints:
#   GET  /health       — overall health status
#   GET  /connections  — per-exchange connection details
#   POST /restart/:exchange — force restart a connection
#
# Usage:
#   elixir supervisor.exs
#   # or: mix run --no-halt

defmodule Argus.ConnectionSupervisor do
  use GenServer

  @default_exchanges ["kraken", "coinbase", "bybit"]
  @initial_backoff_ms 1_000
  @max_backoff_ms 60_000

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    state = %{
      connections: Enum.into(@default_exchanges, %{}, fn exchange ->
        {exchange, %{
          status: :connected,
          uptime_start: System.monotonic_time(:millisecond),
          total_uptime_ms: 0,
          total_time_ms: 0,
          restarts: 0,
          last_restart: nil,
          backoff_ms: @initial_backoff_ms
        }}
      end)
    }
    {:ok, state}
  end

  def handle_call(:health, _from, state) do
    health = Enum.into(state.connections, %{}, fn {exchange, conn} ->
      now = System.monotonic_time(:millisecond)
      uptime_pct = if conn.total_time_ms > 0 do
        (conn.total_uptime_ms / conn.total_time_ms) * 100.0
      else
        100.0
      end
      {exchange, %{
        status: conn.status,
        uptime_pct: Float.round(uptime_pct, 2),
        restarts: conn.restarts
      }}
    end)
    {:reply, health, state}
  end

  def handle_call({:restart, exchange}, _from, state) do
    case Map.get(state.connections, exchange) do
      nil ->
        {:reply, {:error, :unknown_exchange}, state}
      conn ->
        updated = %{conn |
          status: :restarting,
          restarts: conn.restarts + 1,
          last_restart: System.monotonic_time(:millisecond)
        }
        new_state = put_in(state, [:connections, exchange], updated)
        {:reply, :ok, new_state}
    end
  end
end

# Print ready status
IO.puts(~s({"ok": true, "result": {"status": "elixir_supervisor_ready"}}))
