% ── Emotion polarity ──────────────────────────────────────────────────────────
positive_emotion(happiness).
positive_emotion(surprise).

negative_emotion(sadness).
negative_emotion(anger).
negative_emotion(fear).
negative_emotion(disgust).

neutral_emotion(neutral).

% ── Emotional intensity ───────────────────────────────────────────────────────
high_intensity(anger).
high_intensity(fear).
high_intensity(disgust).

% ── Inter-modality relationship classification ────────────────────────────────

% Uncertain — both confidences too low to trust anything
modality_case(_, _, VC, TC, uncertain) :-
    VC < 0.4, TC < 0.4, !.

% Masking — sadness face + positive text
modality_case(sadness, TE, _, _, masking) :-
    positive_emotion(TE), !.

% Irony/Sarcasm — high intensity negative face + positive text
%    Anger, fear, disgust with positive words = sarcasm signal.
modality_case(VE, TE, _, _, irony) :-
    negative_emotion(VE),
    positive_emotion(TE), !.

% Full agreement — both predict same emotion
modality_case(E, E, _, _, agreement) :- !.

% Masking — any negative face + neutral text
modality_case(VE, TE, _, _, masking) :-
    negative_emotion(VE),
    neutral_emotion(TE), !.

% Neutral override — one modality predicts neutral, other predicts specific
% Vision neutral, text specific → trust text
modality_case(VE, TE, _, _, neutral_override) :-
    neutral_emotion(VE),
    \+ neutral_emotion(TE), !.

% Text neutral, vision specific → trust vision  
modality_case(VE, TE, _, _, neutral_override) :-
    neutral_emotion(TE),
    \+ neutral_emotion(VE), !.

% Partial agreement — different emotions but same polarity
modality_case(VE, TE, _, _, partial) :-
    positive_emotion(VE), positive_emotion(TE), !.
modality_case(VE, TE, _, _, partial) :-
    negative_emotion(VE), negative_emotion(TE), !.

% Conflict — face positive, text negative
modality_case(VE, TE, _, _, conflict) :-
    positive_emotion(VE),
    negative_emotion(TE), !.

% Catch-all
modality_case(_, _, _, _, uncertain).

% ── Dominant emotion resolution ───────────────────────────────────────────────

% Uncertain → default to neutral
resolve(_, _, _, _, uncertain, neutral, low) :- !.

% Irony → trust vision (face does not lie), medium confidence
resolve(VE, _TE, _, _, irony, VE, medium) :- !.

% Agreement → use the shared emotion, high confidence
resolve(E, E, _, _, agreement, E, high) :- !.

% Masking → trust vision regardless of confidence
resolve(VE, _TE, _, _, masking, VE, medium) :- !.

% Neutral override → trust the non-neutral modality unconditionally
resolve(VE, TE, _, _, neutral_override, TE, medium) :-
    neutral_emotion(VE), !.   % vision is neutral → trust text
resolve(VE, _TE, _, _, neutral_override, VE, medium) :- !.  % text is neutral → trust vision

% Partial agreement → trust higher confidence modality
resolve(VE, _TE, VC, TC, partial, VE, medium) :- VC >= TC, !.
resolve(_VE, TE, _VC, _TC, partial, TE, medium) :- !.

% Conflict → trust higher confidence modality
resolve(VE, _TE, VC, TC, conflict, VE, medium) :- VC >= TC, !.
resolve(_VE, TE, _VC, _TC, conflict, TE, medium) :- !.

% ── Response strategy ─────────────────────────────────────────────────────────

% Irony detected — special strategy regardless of confidence
% Handled at top level before generic strategy rules
response_strategy(_, _, irony, irony_aware) :- !.

% High intensity negative with high confidence → empathetic priority
response_strategy(E, high, _, empathetic_priority) :-
    high_intensity(E), negative_emotion(E), !.

% Any negative with high confidence → acknowledge and adapt
response_strategy(E, high, _, acknowledge_and_adapt) :-
    negative_emotion(E), !.

% Positive with high confidence → reinforce
response_strategy(E, high, _, reinforce_positive) :-
    positive_emotion(E), !.

% Neutral high confidence → neutral supportive
response_strategy(neutral, high, _, neutral_supportive) :- !.

% Neutral override with medium — use gentle acknowledgement
% since we are overriding a neutral signal, we are less certain
% response_strategy(E, neutral_override_conf, _, gentle_acknowledgement) :-
%    negative_emotion(E), !.

% Medium confidence negative → gentle acknowledgement
response_strategy(E, medium, _, gentle_acknowledgement) :-
    negative_emotion(E), !.

% Everything else → neutral supportive
response_strategy(_, _, _, neutral_supportive).

% ── Top-level entry point ─────────────────────────────────────────────────────
% emotion_agent(+VisionEmotion, +TextEmotion, +VisionConf, +TextConf,
%               -Strategy, -DominantEmotion, -ConfidenceLevel, -Case)

emotion_agent(VE, TE, VC, TC, Strategy, DominantEmotion, ConfidenceLevel, Case) :-
    modality_case(VE, TE, VC, TC, Case),
    resolve(VE, TE, VC, TC, Case, DominantEmotion, ConfidenceLevel),
    response_strategy(DominantEmotion, ConfidenceLevel, Case, Strategy).