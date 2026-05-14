%% @doc Simple JSON encoder/decoder for the Argus trading system.
%% Encoder  : Erlang maps/lists/atoms/numbers  -> JSON binary.
%% Decoder  : recursive-descent parser for JSON -> Erlang terms.
-module(argus_json).
-export([encode/1, decode/1]).

%%====================================================================
%% Public API
%%====================================================================

encode(true)      -> <<"true">>;
encode(false)     -> <<"false">>;
encode(null)      -> <<"null">>;
encode(undefined) -> <<"null">>;
encode(X) when is_integer(X) ->
    list_to_binary(integer_to_list(X));
encode(X) when is_float(X) ->
    format_float(X);
encode(X) when is_atom(X) ->
    encode(atom_to_binary(X, utf8));
encode(X) when is_binary(X) ->
    <<34, (escape_string(X))/binary, 34>>;
encode(X) when is_list(X) ->
    case is_proplist(X) of
        true  -> encode_object(X);
        false -> encode_array(X)
    end;
encode(X) when is_map(X) ->
    encode_map(X);
encode(_) -> <<"null">>.

decode(Bin) when is_binary(Bin) ->
    decode(binary_to_list(Bin));
decode(Str) when is_list(Str) ->
    {Value, _Rest} = parse_value(skip_ws(Str)),
    Value.

%%====================================================================
%% Encoder internals
%%====================================================================

format_float(X) ->
    Str = lists:flatten(io_lib:format("~.10g", [X])),
    list_to_binary(Str).

escape_string(Bin) -> escape_string(Bin, <<>>).

escape_string(<<>>, Acc) -> Acc;
escape_string(<<34, Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, 92, 34>>);
escape_string(<<92, Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, 92, 92>>);
escape_string(<<10, Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, 92, >>);
escape_string(<<13, Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, 92, >>);
escape_string(<<9,  Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, 92, >>);
escape_string(<<C, Rest/binary>>, Acc) ->
    escape_string(Rest, <<Acc/binary, C>>).

encode_array(List) ->
    Items  = [encode(V) || V <- List],
    Joined = bin_join(Items, <<",">>),
    <<"[", Joined/binary, "]">>.

encode_object(Proplist) ->
    Pairs  = [encode_pair(K, V) || {K, V} <- Proplist],
    Joined = bin_join(Pairs, <<",">>),
    <<"{", Joined/binary, "}">>.

encode_map(Map) ->
    Pairs  = [encode_pair(K, V) || {K, V} <- maps:to_list(Map)],
    Joined = bin_join(Pairs, <<",">>),
    <<"{", Joined/binary, "}">>.

encode_pair(K, V) when is_atom(K)    -> encode_pair(atom_to_binary(K, utf8), V);
encode_pair(K, V) when is_integer(K) -> encode_pair(integer_to_binary(K), V);
encode_pair(K, V) when is_list(K)    -> encode_pair(list_to_binary(K), V);
encode_pair(K, V) when is_binary(K)  ->
    KEnc = encode(K),
    VEnc = encode(V),
    <<KEnc/binary, ":", VEnc/binary>>.

is_proplist([])              -> true;
is_proplist([{K, _} | Rest]) when is_atom(K); is_binary(K); is_list(K) ->
    is_proplist(Rest);
is_proplist(_)               -> false.

bin_join([], _Sep)    -> <<>>;
bin_join([H], _Sep)   -> H;
bin_join([H | T], Sep) ->
    lists:foldl(fun(X, Acc) -> <<Acc/binary, Sep/binary, X/binary>> end, H, T).

%%====================================================================
%% Decoder internals
%%====================================================================

skip_ws([C | Rest]) when C =:= 32; C =:= 9; C =:= 10; C =:= 13 ->
    skip_ws(Rest);
skip_ws(Str) -> Str.

parse_value([123 | Rest])                        -> parse_object(skip_ws(Rest), #{});
parse_value([91  | Rest])                        -> parse_array(skip_ws(Rest), []);
parse_value([34  | Rest])                        -> parse_string(Rest, []);
parse_value([, , ,  | Rest])             -> {true,  Rest};
parse_value([, , , ,  | Rest])         -> {false, Rest};
parse_value([, , ,  | Rest])             -> {null,  Rest};
parse_value([C | _] = Str) when C =:= 45 orelse (C >= 48 andalso C =< 57) ->
    parse_number(Str, []);
parse_value([]) -> {null, []}.

parse_object([125 | Rest], Acc) -> {Acc, Rest};
parse_object([34  | Rest], Acc) ->
    {Key, S2} = parse_string(Rest, []),
    [58 | S3] = skip_ws(S2),
    {Val, S4} = parse_value(skip_ws(S3)),
    Acc2      = maps:put(Key, Val, Acc),
    case skip_ws(S4) of
        [44 | S5]  -> parse_object(skip_ws(S5), Acc2);
        [125 | S5] -> {Acc2, S5};
        S5         -> {Acc2, S5}
    end;
parse_object(Str, Acc) -> {Acc, Str}.

parse_array([93 | Rest], Acc) -> {lists:reverse(Acc), Rest};
parse_array(Str, Acc) ->
    {Val, S2} = parse_value(Str),
    case skip_ws(S2) of
        [44 | S3] -> parse_array(skip_ws(S3), [Val | Acc]);
        [93 | S3] -> {lists:reverse([Val | Acc]), S3};
        S3        -> {lists:reverse([Val | Acc]), S3}
    end.

parse_string([], Acc) ->
    {list_to_binary(lists:reverse(Acc)), []};
parse_string([34 | Rest], Acc) ->
    {list_to_binary(lists:reverse(Acc)), Rest};
parse_string([92, 34  | Rest], Acc) -> parse_string(Rest, [34  | Acc]);
parse_string([92, 92  | Rest], Acc) -> parse_string(Rest, [92  | Acc]);
parse_string([92,   | Rest], Acc) -> parse_string(Rest, [10  | Acc]);
parse_string([92,   | Rest], Acc) -> parse_string(Rest, [13  | Acc]);
parse_string([92,   | Rest], Acc) -> parse_string(Rest, [9   | Acc]);
parse_string([92, 47  | Rest], Acc) -> parse_string(Rest, [47  | Acc]);
parse_string([C   | Rest], Acc)     -> parse_string(Rest, [C   | Acc]).

parse_number([C | Rest], Acc)
    when C >= 48, C =< 57; C =:= 45; C =:= 46; C =:= 101; C =:= 69; C =:= 43 ->
    parse_number(Rest, [C | Acc]);
parse_number(Rest, Acc) ->
    {to_number(lists:reverse(Acc)), Rest}.

to_number(Str) ->
    IsFloat = lists:any(
        fun(C) -> C =:= 46 orelse C =:= 101 orelse C =:= 69 end, Str),
    if IsFloat ->
        try list_to_float(Str)
        catch _:_ ->
            try float(list_to_integer(Str))
            catch _:_ -> 0.0 end
        end;
    true ->
        try list_to_integer(Str)
        catch _:_ ->
            try list_to_float(Str)
            catch _:_ -> 0 end
        end
    end.