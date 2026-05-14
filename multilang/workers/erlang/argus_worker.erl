#!/usr/bin/env escript
%% -*- erlang -*-
%%
%% Argus Ultimate -- Erlang multilang worker
%% Profile: language=erlang, risk_max=0.44, cycle_scale=0.99,
%%          vol_w=1.0, sig_w=1.01, spread=1.02
%%
%% Run with:  escript argus_worker.erl
%% Protocol:
%%   stdin  -> {"task_type": "...", "data": {...}}
%%   stdout <- {"ok": true, "result": {...}, "took_ms": 0.12}
%%
%% OTP 27+: uses the built-in json module.
%% OTP < 27: falls back to a hand-rolled JSON parser/encoder.

-mode(compile).

main(_) ->
    %% Use raw port for stdin to avoid BEAM io-server hangs on piped Windows stdin
    Port = open_port({fd, 0, 2}, [in, binary, {line, 65536}]),
    loop(Port).

loop(Port) ->
    receive
        {Port, {data, {eol, Line}}} ->
            Trimmed = string:trim(Line),
            case Trimmed of
                <<>> -> loop(Port);
                _ ->
                    T0     = erlang:monotonic_time(microsecond),
                    Dec    = safe_decode(Trimmed),
                    Res    = dispatch(Dec),
                    TookMs = (erlang:monotonic_time(microsecond) - T0) / 1000.0,
                    Out    = safe_encode(Res#{took_ms => TookMs}),
                    io:put_chars([Out, $\n]),
                    loop(Port)
            end;
        {Port, {data, {noeol, _}}} ->
            loop(Port);
        {Port, eof} ->
            ok
    after 60000 ->
            ok
    end.

safe_decode(Input) ->
    Bin = to_binary(Input),
    try json:decode(Bin)
    catch error:undef -> manual_decode(Bin)
    end.

safe_encode(Value) ->
    try json:encode(Value)
    catch error:undef -> manual_encode(Value)
    end.

to_binary(B) when is_binary(B) -> B;
to_binary(L) when is_list(L)   -> list_to_binary(L).

%% -------------------------------------------------------------------
%% Manual JSON decoder
%% -------------------------------------------------------------------

manual_decode(Bin) ->
    Str   = binary_to_list(Bin),
    Pairs = extract_pairs(Str),
    maps:from_list([{list_to_binary(K), V} || {K, V} <- Pairs]).

extract_pairs(Str) ->
    Inner = string:trim(string:trim(Str, leading, "{"), trailing, "}"),
    split_pairs(Inner, []).

split_pairs([], Acc) -> lists:reverse(Acc);
split_pairs(Str, Acc) ->
    case parse_string(string:trim(Str)) of
        {Key, Rest1} ->
            Rest2 = string:trim(string:trim(Rest1), leading, ":"),
            {Val, Rest3} = parse_value(string:trim(Rest2)),
            Rest4 = string:trim(string:trim(Rest3), leading, ","),
            split_pairs(Rest4, [{Key, Val} | Acc]);
        error ->
            lists:reverse(Acc)
    end.

parse_string([$" | Rest]) -> parse_string_body(Rest, []);
parse_string(_)           -> error.

parse_string_body([$" | Rest], Acc)        -> {lists:reverse(Acc), Rest};
parse_string_body([$\\, C | Rest], Acc)   -> parse_string_body(Rest, [C | Acc]);
parse_string_body([C | Rest], Acc)         -> parse_string_body(Rest, [C | Acc]);
parse_string_body([], Acc)                 -> {lists:reverse(Acc), []}.

parse_value([${ | _] = Str) ->
    {ObjStr, Rest} = collect_balanced(Str, ${, $}),
    {manual_decode(list_to_binary(ObjStr)), Rest};
parse_value([$[ | _] = Str) ->
    {ArrStr, Rest} = collect_balanced(Str, $[, $]),
    {parse_array(ArrStr), Rest};
parse_value([$" | Rest]) ->
    case parse_string_body(Rest, []) of
        {S, R} -> {list_to_binary(S), R};
        error  -> {<<>>, Rest}
    end;
parse_value("true"  ++ Rest) -> {true,  Rest};
parse_value("false" ++ Rest) -> {false, Rest};
parse_value("null"  ++ Rest) -> {null,  Rest};
parse_value(Str) ->
    {NumStr, Rest} = split_number(Str, []),
    case string:to_float(NumStr) of
        {F, _} -> {F, Rest};
        _ ->
            case string:to_integer(NumStr) of
                {I, _} -> {I, Rest};
                _      -> {null, Rest}
            end
    end.

split_number([], Acc)          -> {lists:reverse(Acc), []};
split_number([C | R] = S, Acc) ->
    case lists:member(C, "0123456789.eE+-") of
        true  -> split_number(R, [C | Acc]);
        false -> {lists:reverse(Acc), S}
    end.

collect_balanced([Open | Rest], Open, Close) ->
    collect_balanced(Rest, Open, Close, 1, [Open]).
collect_balanced([], _O, _C, _, Acc) ->
    {lists:reverse(Acc), []};
collect_balanced([C | Rest], Open, Close, D, Acc) when C =:= Close ->
    NA = [C | Acc],
    if D =:= 1 -> {lists:reverse(NA), Rest};
       true    -> collect_balanced(Rest, Open, Close, D - 1, NA)
    end;
collect_balanced([C | Rest], Open, Close, D, Acc) when C =:= Open ->
    collect_balanced(Rest, Open, Close, D + 1, [C | Acc]);
collect_balanced([C | Rest], Open, Close, D, Acc) ->
    collect_balanced(Rest, Open, Close, D, [C | Acc]).

parse_array(Str) ->
    Inner = string:trim(string:trim(Str, leading, "["), trailing, "]"),
    parse_array_items(string:trim(Inner), []).

parse_array_items([], Acc) -> lists:reverse(Acc);
parse_array_items(Str, Acc) ->
    {Val, Rest} = parse_value(Str),
    Rest2 = string:trim(string:trim(Rest), leading, ","),
    parse_array_items(Rest2, [Val | Acc]).

%% -------------------------------------------------------------------
%% Manual JSON encoder
%% -------------------------------------------------------------------

manual_encode(M) when is_map(M) ->
    Pairs = maps:to_list(M),
    Inner = lists:join(",", [enc_pair(K, V) || {K, V} <- Pairs]),
    ["{", Inner, "}"];
manual_encode(L) when is_list(L) ->
    ["[", lists:join(",", [manual_encode(E) || E <- L]), "]"];
manual_encode(true)  -> "true";
manual_encode(false) -> "false";
manual_encode(null)  -> "null";
manual_encode(B) when is_binary(B) ->
    [$", esc(binary_to_list(B)), $"];
manual_encode(A) when is_atom(A) ->
    [$", atom_to_list(A), $"];
manual_encode(F) when is_float(F)   -> io_lib:format("~f", [F]);
manual_encode(I) when is_integer(I) -> integer_to_list(I).

enc_pair(K, V) when is_binary(K) -> [$", esc(binary_to_list(K)), $", ":", manual_encode(V)];
enc_pair(K, V) when is_atom(K)   -> [$", atom_to_list(K), $", ":", manual_encode(V)].

esc([])         -> [];
esc([$" | R])   -> [$\\, $" | esc(R)];
esc([$\\ | R])  -> [$\\, $\\ | esc(R)];
esc([$\n | R])  -> [$\\, $n | esc(R)];
esc([$\r | R])  -> [$\\, $r | esc(R)];
esc([$\t | R])  -> [$\\, $t | esc(R)];
esc([C | R])    -> [C | esc(R)].

%% ====================================================================
%% Profile
%% ====================================================================

-define(RISK_MAX,    0.44).
-define(CYCLE_SCALE, 0.99).
-define(VOL_W,       1.0).
-define(SIG_W,       1.01).
-define(SPREAD,      1.02).

%% ====================================================================
%% Dispatch
%% ====================================================================

dispatch(#{<<"task_type">> := TT} = Msg) ->
    Data = maps:get(<<"data">>, Msg, #{}),
    handle(TT, Data);
dispatch(_) ->
    #{ok => false, error => <<"missing task_type">>}.

handle(<<"heartbeat">>, _) ->
    #{ok => true, language => <<"erlang">>, ts => erlang:system_time(millisecond)};

handle(<<"cycle_plan">>, Data) ->
    H  = cycle_hash(Data),
    AC = pick_action(H),
    CF = 0.5 + (H rem 50) / 100.0,
    SZ = ?CYCLE_SCALE * (0.1 + (H rem 10) / 100.0),
    #{ok => true, language => <<"erlang">>, cycle_boost => H,
      action => AC, confidence => CF, size => SZ};

handle(<<"volatility_estimate">>, Data) ->
    Vol = compute_vol(get_prices(Data)),
    #{ok => true, language => <<"erlang">>, volatility_annual_bps => Vol * ?VOL_W, volatility_weight => 1.0};

handle(<<"signal_score">>, Data) ->
    H     = cycle_hash(Data),
    Delta = ((H rem 100) - 50) / 5000.0 * ?SIG_W,
    #{ok => true, language => <<"erlang">>, score_delta => Delta, signal_score_weight => 1.0};

handle(<<"risk_calculation">>, Data) ->
    PV = get_float(<<"position_value">>, Data, 0.0),
    CA = get_float(<<"capital">>,        Data, 1.0),
    C2 = if CA == 0 -> 1.0; true -> CA end,
    RR = PV / C2,
    #{ok => true, language => <<"erlang">>, exposure_ratio => RR,
      passed => RR =< ?RISK_MAX, max_ratio => ?RISK_MAX};

handle(<<"position_sizing">>, Data) ->
    CA  = get_float(<<"capital">>,   Data, 10000.0),
    RP  = get_float(<<"risk_pct">>,  Data, 0.01),
    SD  = get_float(<<"stop_dist">>, Data, 1.0),
    SD2 = if SD == 0 -> 1.0; true -> SD end,
    RawSize = (CA * RP) / SD2,
    #{ok => true, language => <<"erlang">>, size_pct => RawSize / CA, size_abs => RawSize};

handle(<<"drawdown_check">>, Data) ->
    PK = get_float(<<"peak">>,    Data, 1.0),
    CU = get_float(<<"current">>, Data, 1.0),
    P2 = if PK == 0 -> 1.0; true -> PK end,
    DD = (P2 - CU) / P2,
    #{ok => true, language => <<"erlang">>, current_drawdown_pct => DD, passed => DD < 0.20};

handle(<<"var_estimate">>, Data) ->
    Vol = compute_vol(get_prices(Data)),
    VaR = Vol * 1.645,
    #{ok => true, language => <<"erlang">>, var_pct => VaR, cvar_pct => VaR * 1.2};

handle(<<"skew_estimate">>, Data) ->
    #{ok => true, language => <<"erlang">>, skew => compute_skew(get_prices(Data))};

handle(<<"order_book_imbalance_series">>, Data) ->
    B = get_float(<<"bid_volume">>, Data, 100.0),
    A = get_float(<<"ask_volume">>, Data, 100.0),
    D = B + A,
    Imb = if D == 0 -> 0.0; true -> (B - A) / D end,
    #{ok => true, language => <<"erlang">>, imbalance_series => [Imb], trend => Imb};

handle(<<"order_book_processing">>, Data) ->
    B = get_float(<<"bid">>, Data, 0.0),
    A = get_float(<<"ask">>, Data, 0.0),
    Sp = if A > 0.0 -> (A - B) / A * ?SPREAD; true -> 0.0 end,
    Mid = (B + A) / 2.0,
    #{ok => true, language => <<"erlang">>, spread_bps => Sp, imbalance => 0.0, mid => Mid};

handle(<<"regime_estimate">>, Data) ->
    Vol    = compute_vol(get_prices(Data)),
    Regime = if Vol > 0.02 -> <<"high_vol">>; true -> <<"low_vol">> end,
    #{ok => true, language => <<"erlang">>, regime => Regime, confidence => 0.5, regime_weight => 1.0};

handle(<<"slippage_estimate">>, Data) ->
    Sz = get_float(<<"size">>,   Data, 1.0),
    Sp = get_float(<<"spread">>, Data, 0.01),
    #{ok => true, language => <<"erlang">>, slippage_bps => Sz * Sp * ?SPREAD};

handle(<<"correlation_estimate">>, _) ->
    #{ok => true, language => <<"erlang">>, correlation => 0.0, value => 0.5};

handle(<<"liquidity_score">>, Data) ->
    Vol   = get_float(<<"volume">>, Data, 1000.0),
    Score = min(1.0, Vol / 10000.0),
    #{ok => true, language => <<"erlang">>, liquidity_score => Score, depth_bps => 100};

handle(<<"market_impact">>, Data) ->
    Sz  = get_float(<<"size">>,      Data, 1.0),
    Liq = get_float(<<"liquidity">>, Data, 1000.0),
    L2  = if Liq == 0 -> 1.0; true -> Liq end,
    #{ok => true, language => <<"erlang">>, impact_bps => Sz / L2};

handle(<<"signal_filter">>, Data) ->
    Sig = get_float(<<"signal">>,    Data, 0.0),
    Thr = get_float(<<"threshold">>, Data, 0.1),
    #{ok => true, language => <<"erlang">>,
      accept => abs(Sig) >= Thr, filter_reason => <<>>, signal => Sig};

handle(<<"confidence_calibration">>, Data) ->
    Raw = get_float(<<"confidence">>, Data, 0.5),
    Cal = max(0.0, min(1.0, Raw)),
    #{ok => true, language => <<"erlang">>, calibrated_confidence => Cal};

handle(<<"execution_quality_score">>, Data) ->
    Slip  = get_float(<<"slippage">>, Data, 0.0),
    Score = max(0.0, 1.0 - abs(Slip) * 10.0),
    #{ok => true, language => <<"erlang">>, score_0_1 => Score, avg_slippage_bps => 0};

handle(<<"regime_duration">>, Data) ->
    Start = get_float(<<"start_ts">>, Data, 0.0),
    Now   = float(erlang:system_time(millisecond)),
    Bars = Now - Start,
    #{ok => true, language => <<"erlang">>, bars_in_regime => Bars, regime_stable => true, regime => <<"unknown">>};

handle(TT, _) ->
    #{ok => true, language => <<"erlang">>, task => TT, value => 0.5}.

%% ====================================================================
%% Helpers
%% ====================================================================

get_float(Key, Map, Def) ->
    case maps:get(Key, Map, Def) of
        V when is_float(V)   -> V;
        V when is_integer(V) -> float(V);
        _                    -> Def
    end.

get_prices(Data) ->
    case maps:get(<<"prices">>, Data, []) of
        L when is_list(L) ->
            [case P of
                 F when is_float(F)   -> F;
                 I when is_integer(I) -> float(I);
                 _                    -> 0.0
             end || P <- L];
        _ -> []
    end.

compute_vol([])     -> 0.0;
compute_vol([_])    -> 0.0;
compute_vol(Prices) -> std_dev(log_returns(Prices)).

log_returns([])              -> [];
log_returns([_])             -> [];
log_returns([P1, P2 | Rest]) ->
    Safe = if P1 == 0 -> 0.0; true -> math:log(P2 / P1) end,
    [Safe | log_returns([P2 | Rest])].

std_dev([]) -> 0.0;
std_dev(Xs) ->
    N    = length(Xs),
    Mean = lists:sum(Xs) / N,
    Var  = lists:sum([(X - Mean) * (X - Mean) || X <- Xs]) / N,
    math:sqrt(Var).

compute_skew([])  -> 0.0;
compute_skew([_]) -> 0.0;
compute_skew(Xs) ->
    N    = length(Xs),
    Mean = lists:sum(Xs) / N,
    Std  = std_dev(Xs),
    if Std == 0 -> 0.0;
       true ->
           S3 = lists:sum([math:pow((X - Mean) / Std, 3) || X <- Xs]),
           S3 / N
    end.

cycle_hash(Data) ->
    Pairs = lists:sort(maps:to_list(Data)),
    Str   = lists:flatten([io_lib:format("~p:~p,", [K, V]) || {K, V} <- Pairs]),
    Bytes = erlang:md5(list_to_binary(Str)),
    binary:decode_unsigned(binary:part(Bytes, 0, 4)).

pick_action(H) ->
    case H rem 3 of
        0 -> <<"buy">>;
        1 -> <<"sell">>;
        _ -> <<"hold">>
    end.
