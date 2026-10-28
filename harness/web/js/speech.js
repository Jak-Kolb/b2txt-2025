// Speak decoded words aloud as they become final, the way the participant's words would be heard.
// Partial outputs are revised as more neural data arrives, so a word is spoken only once it is committed by the
// live-word rule used in the clock-A evaluations (learning/live_eval/clock_a.py, "sil+D"): word n is committed D
// frames (80 ms each) after the model's own n-th greedy word-boundary (" | ") emission, as it reads in the partial
// at that frame. Words are spoken once, in order; what is still unspoken at the trial's final output is spoken then.
// Spoken words cannot be taken back, so later revisions of them are live errors, as in the evaluations.

export class WordSpeaker {
  // say(text): speak one utterance. delay: D in frames, or "final" to speak the whole final sentence at the end.
  constructor(say, delay = 1) {
    this.say = say;
    this.delay = delay;
    this.reset();
  }

  reset() {
    this.boundaries = [];      // frame index of each greedy word-boundary emission
    this.lastPhone = null;     // previous frame's greedy class (null = blank), for CTC collapsing
    this.sawPhoneme = false;   // a phoneme was emitted since the last boundary (clock_a.word_boundaries)
    this.spoken = [];          // words spoken so far, in order
  }

  speak(words) {
    if (!words.length) return;
    this.spoken.push(...words);
    this.say(words.join(" "));
  }

  // One live frame event: { frame, phone, words }.
  frame({ frame, phone, words }) {
    if (phone !== this.lastPhone) {
      if (phone === "|" && this.sawPhoneme) { this.boundaries.push(frame); this.sawPhoneme = false; }
      else if (phone && phone !== "|") this.sawPhoneme = true;
    }
    this.lastPhone = phone;
    if (this.delay === "final") return;
    let n = this.spoken.length;
    while (n < this.boundaries.length && frame >= this.boundaries[n] + this.delay) n++;
    // commit through word n (as many of them as the partial shows now)
    this.speak(words.slice(this.spoken.length, Math.min(n, words.length)));
  }

  // The trial's final output.
  final(words) {
    this.speak(words.slice(this.spoken.length));
  }
}
