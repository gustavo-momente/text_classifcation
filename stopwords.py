#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Usuário'


class FRStopWords:
    def __init__(self):
        self.toplist = set('"alors au aucuns aussi autre avant avec avoir bon car ce cela ces ceux chaque ci comme '
                           'comment dans des du dedans dehors depuis deux devrait doit donc dos droite début elle '
                           'elles en encore essai est et eu fait faites fois font force haut hors ici il ils je	juste '
                           'la le les leur là ma maintenant mais mes mine moins mon mot même ni nommés notre nous '
                           'nouveaux ou où par parce parole pas personnes peut peu pièce plupart pour pourquoi quand '
                           'que quel quelle quelles quels qui sa sans ses seulement si sien son sont sous soyez sujet '
                           'sur ta tandis tellement tels tes ton tous tout trop très tu valeur voie voient vont votre '
                           'vous vu ça étaient état étions été être de à un une ne d l c s je"'.split())

    def get_toplist(self):
        return self.toplist


class ENStopWords:
    def __init__(self):
        self.toplist = set("a about above after again against all am an and any are aren as at be because been before "
                           "being below between both but by can cannot could couldn did didn do does doesn doing don "
                           "down during each few for from further had hadn has hasn have haven having he her here hers "
                           "herself him himself his how i if in into is isn it its itself let me more most mustn my "
                           "myself no nor not of off on once only or other ought our ours ourselves out over own same "
                           "shan she should shouldn so some such than that the their theirs them themselves then there "
                           "these they this those through to too under until up very was wasn we were weren what when "
                           "where which while who whom why with won would wouldn you your yours yourself yourselves "
                           "ll de ve s".split())

    def get_toplist(self):
        return self.toplist
