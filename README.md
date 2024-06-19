# KI-Praktikum-Baran-Fischer-Wewer

Idee: Die Idee dieses Projektes ist ein Satelitenbild zu nehmen und mithilfe von GANs ein Satelitenbild zu generieren auf dem der Ausbruch eines Feuers simuliert wird.

Schwierigkeit 1: Die Datensammlung stellt eine erhebliche Herausforderung da, denn es gibt nur wenige und nicht ausreichende Bilder die für unseren Ansatz genutzt werden können. Aus diesem Grund war der erste Ansatz das Datenset https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset zu verwenden. Das Problem bei diesem Datenset ist allerdings das die Bilder von Waldbränden aus anderen Koordinaten kommen als die Bilder ohne Waldbrände. Daher kann dieses Datenset nicht genutzt werden um den Ausbruch eines Feuers  vorher und nachher darzustellen. Deswegen war der erste Ansatz ein CycleGAN zu verwenden.

CycleGAN Ansatz Ersan



Ansatz Satelitendaten per Bot zum Datenset zu ergänzen Dennis:
Da dieser Ansatz gescheitert ist wurde die Entscheidung getroffen das für jedes Waldbrand Bild aus dem Datenset ein äquivalentes Bild der selben Koordinaten ohne Waldbrand her muss. Dafür war der erste Ansatz dies mithilfe eines python scripts durchzuführen und für jede Koordinate mit Waldbrand ein Satelitenbild herunterzuladen. Dieser Ansatz ist leider gescheitert da alle Websites die getestet wurden nach ca. 10 bis 20 aufrufen der Website keine Anfragen mehr ausgegeben haben. Deswegen war die nächste Idee die Satelitenbilder mithilfe einer API von Google Earth Images herunterzuladen.


Datensammlung und Verarbeitung Sebastian:
Dieser Ansatz hat trotz einiger komplikationen sehr gut funktioniert und so existiert nun vollständiges Datenset aus den selben Koordinaten mit und ohne Waldbrand.

Da wir nun ein vollständiges Datenset haben besteht die Möglichkeit weitere GAN Ansätze zu testen.

DeepGAN Ansatz Alle:


