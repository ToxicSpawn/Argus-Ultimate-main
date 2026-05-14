// ARGUS Go Order Router
//
// High-performance HTTP order routing service.
//
// Endpoints:
//   POST /route   — select fastest venue for an order
//   POST /submit  — submit order to selected venue (concurrent)
//   POST /status  — return latency stats per venue
//   GET  /health  — liveness probe
//
// Build:
//   go build -o go_router .
//
// Run:
//   ./go_router              # default :9998
//   ./go_router -port 9998

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

// ── Types ────────────────────────────────────────────────────────────────────

type Order struct {
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Quantity float64 `json:"quantity"`
	Price    float64 `json:"price,omitempty"`
	Type     string  `json:"type"` // limit, market
}

type VenueStats struct {
	Name       string  `json:"name"`
	AvgLatency float64 `json:"avg_latency_ms"`
	P99Latency float64 `json:"p99_latency_ms"`
	Healthy    bool    `json:"healthy"`
	LastCheck  string  `json:"last_check"`
	FillRate   float64 `json:"fill_rate"`
}

type RouteRequest struct {
	Order  Order    `json:"order"`
	Venues []string `json:"venues,omitempty"`
}

type RouteResponse struct {
	Venue     string  `json:"venue"`
	Score     float64 `json:"score"`
	LatencyMs float64 `json:"latency_ms"`
	Reason    string  `json:"reason"`
}

type SubmitResponse struct {
	Venue    string `json:"venue"`
	OrderID  string `json:"order_id"`
	Status   string `json:"status"`
	LatencyMs float64 `json:"latency_ms"`
}

// ── Router ───────────────────────────────────────────────────────────────────

type Router struct {
	mu       sync.RWMutex
	venues   map[string]*venueState
}

type venueState struct {
	name      string
	latencies []float64
	healthy   bool
	lastCheck time.Time
	fillRate  float64
}

func NewRouter() *Router {
	r := &Router{
		venues: make(map[string]*venueState),
	}
	// Default venues
	for _, name := range []string{"kraken", "coinbase", "binance", "bybit"} {
		r.venues[name] = &venueState{
			name:      name,
			latencies: []float64{50, 60, 55, 52, 48},
			healthy:   true,
			lastCheck: time.Now(),
			fillRate:  0.95,
		}
	}
	return r
}

func (r *Router) Route(order Order, preferredVenues []string) RouteResponse {
	r.mu.RLock()
	defer r.mu.RUnlock()

	candidates := r.venues
	if len(preferredVenues) > 0 {
		candidates = make(map[string]*venueState)
		for _, v := range preferredVenues {
			if vs, ok := r.venues[v]; ok {
				candidates[v] = vs
			}
		}
	}

	var best string
	bestScore := math.Inf(1)

	for name, vs := range candidates {
		if !vs.healthy {
			continue
		}
		avgLat := avg(vs.latencies)
		// Score: lower latency + higher fill rate = better
		score := avgLat * (2.0 - vs.fillRate)
		if score < bestScore {
			bestScore = score
			best = name
		}
	}

	if best == "" {
		// All unhealthy — pick first available
		for name := range candidates {
			best = name
			break
		}
		if best == "" {
			best = "kraken" // ultimate fallback
		}
	}

	vs := r.venues[best]
	return RouteResponse{
		Venue:     best,
		Score:     bestScore,
		LatencyMs: avg(vs.latencies),
		Reason:    "lowest_score",
	}
}

func (r *Router) Submit(order Order, venue string) SubmitResponse {
	start := time.Now()
	// Simulate order submission (in production, this calls exchange API)
	time.Sleep(1 * time.Millisecond) // minimal simulated latency

	r.mu.Lock()
	if vs, ok := r.venues[venue]; ok {
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		vs.latencies = append(vs.latencies, elapsed)
		if len(vs.latencies) > 100 {
			vs.latencies = vs.latencies[len(vs.latencies)-100:]
		}
	}
	r.mu.Unlock()

	return SubmitResponse{
		Venue:     venue,
		OrderID:   fmt.Sprintf("sim_%d", time.Now().UnixNano()),
		Status:    "submitted",
		LatencyMs: float64(time.Since(start).Microseconds()) / 1000.0,
	}
}

func (r *Router) GetStats() []VenueStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := make([]VenueStats, 0, len(r.venues))
	for _, vs := range r.venues {
		stats = append(stats, VenueStats{
			Name:       vs.name,
			AvgLatency: avg(vs.latencies),
			P99Latency: percentile(vs.latencies, 99),
			Healthy:    vs.healthy,
			LastCheck:  vs.lastCheck.Format(time.RFC3339),
			FillRate:   vs.fillRate,
		})
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].Name < stats[j].Name })
	return stats
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func avg(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func percentile(vals []float64, pct float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	idx := int(float64(len(sorted)-1) * pct / 100.0)
	return sorted[idx]
}

// ── HTTP Handlers ────────────────────────────────────────────────────────────

func main() {
	port := "9998"
	if len(os.Args) > 2 && os.Args[1] == "-port" {
		port = os.Args[2]
	}

	router := NewRouter()

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	http.HandleFunc("/route", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req RouteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := router.Route(req.Order, req.Venues)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/submit", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Order Order  `json:"order"`
			Venue string `json:"venue"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Venue == "" {
			route := router.Route(req.Order, nil)
			req.Venue = route.Venue
		}
		resp := router.Submit(req.Order, req.Venue)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost && r.Method != http.MethodGet {
			http.Error(w, "POST or GET", http.StatusMethodNotAllowed)
			return
		}
		stats := router.GetStats()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})

	addr := fmt.Sprintf("127.0.0.1:%s", port)
	log.Printf("[GO_ROUTER] Listening on http://%s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
