
tags_all = '''#tags
#abstractthinking
#adaptation
#affect
#altruism
#analysis
#biology
#book
#care
#classdivision
#communication
#control
#crucial
#crucialpoint
#cybernetics
#description
#differentfields
#DonellMeadows
#draft
#egoism
#entropy
#example
#extensionality
#flexibility
#flexiblemindset
#holism
#ideas
#ideology
#information
#material
#mechanism
#metaphor
#mind
#money
#multiagentsystem
#paper
#paradigm
#people
#pointofview
#publicopinion
#quote
#reality
#reference
#relationships
#rules
#signal
#sosiocybernetics
#spectral
#study
#surviving
#synchronisation
#systemsanalysis
#systemsborders
#systemsthiking
#terminology
#wiener'''

tags_new = "#notes #quotes #example #reference #paper #study #material #insights #ideas #hypothesis #synergy #approach #systemicapproaches #comparison #organization #strategy #planing #simulation #software #simulationsoftware #strategicplaning #control #agile #adaptive #communication #within #withinsystem #decision #decisionmaking #informed #informeddecisionmaking #dataanalytics #clarify #notclarify #andsoon #paradigm #discovery #explanation #abstraction #abstractionlevels #mindset #systemic #systemicmindset #everchanging #world #everchangingworld #network #thinking #networkthinking #behaviour #economics #behavioraleconomics"

tags_all = list(map(lambda x: x.replace("#", ""), tags_all.split("\n")))
tags_new = list(map(lambda x: x.replace("#", ""), tags_new.split(" ")))

def refresh_tags(tags_all, tags_new):
    tags_list = sorted(list(set(tags_all + tags_new)), key=lambda x: x.lower())
    tags_string = "".join(list(map(lambda x: "#" + x + "\n", tags_list)))
    return tags_list, tags_string


tags_list, tags_string = refresh_tags(tags_all, tags_new)

# print(tags_string)


