# from midiutil.MidiFile import MIDIFile
#
# # create your MIDI object
# mf = MIDIFile(1)     # only 1 track
# track = 0   # the only track
#
# time = 0    # start at the beginning
# mf.addTrackName(track, time, "Sample Track")
# mf.addTempo(track, time, 120)
#
#
# NOTES_NUMBER = 128
# BEAT_STEP = 2
#
# channel = 10
# volume = 100
#
# # notes = {
# #          "track": [0] * NOTES_NUMBER,
# #          "channel": [0] * NOTES_NUMBER,
# #          "pitch": list(range(NOTES_NUMBER)), # MIDI note
# #          "time": list(range(0, BEAT_STEP * NOTES_NUMBER, BEAT_STEP)), # start on beat
# #          "duration": [0] * NOTES_NUMBER, # 1 beat long
# #          "volume": [100] * NOTES_NUMBER,
# #         }
#
# # add_notes_to_track
# for n in range(NOTES_NUMBER):
#     mf.addNote(track=track,
#                channel=channel,
#                pitch=n,
#                time=(2 * n),
#                duration=1,
#                volume=100)
#
# # write it to disk
# with open("output.mid", 'wb') as outf:
#     mf.writeFile(outf)


from miditime.miditime import MIDITime

# Instantiate the class with a tempo (120bpm is the default) and an output file destination.
mymidi = MIDITime(120, 'myfile.mid')

# Create a list of notes. Each note is a list: [time, pitch, velocity, duration]
# midinotes = [
#     [0, 60, 127, 3],  #At 0 beats (the start), Middle C with velocity 127, for 3 beats
#     [10, 61, 127, 4]  #At 10 beats (12 seconds from start), C#5 with velocity 127, for 4 beats
# ]

NOTES_NUMBER = 128

midinotes = []
for n in range(NOTES_NUMBER):
    midinotes.append([2*n, n, 127, 4])

# Add a track with those notes
mymidi.add_track(midinotes)

# Output the .mid file
mymidi.save_midi()
