# KI-Praktikum-Baran-Fischer-Wewer

Idee: In diesem Projekt wollen wir die mögliche Ausbreitung von Waldbränden unter Zuhilfenahme von GANs simulieren. Hierfür verwenden wir echte Satellitenbilder mit und ohne Waldbrände, um diverse GANs zu trainieren und im Anschluss unser Ergebnis und die Fähigkeit unseres Modells realistische Satellitenbilder von Bränden zu erzeugen zu prüfen.

Schwierigkeit 1: Die Datensammlung stellt eine erhebliche Herausforderung da, denn es gibt nur wenige und nicht ausreichende Bilder die für unseren Ansatz genutzt werden können. Aus diesem Grund war der erste Ansatz das Datenset https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset zu verwenden. Das Problem bei diesem Datenset ist allerdings das die Bilder von Waldbränden aus anderen Koordinaten kommen als die Bilder ohne Waldbrände. Daher kann dieses Datenset nicht genutzt werden um den Ausbruch eines Feuers  vorher und nachher darzustellen. Deswegen war der erste Ansatz ein CycleGAN zu verwenden.

CycleGAN Ansatz Ersan



Ansatz Satelitendaten per Bot zum Datenset zu ergänzen Dennis:
Da dieser Ansatz gescheitert ist wurde die Entscheidung getroffen das für jedes Waldbrand Bild aus dem Datenset ein äquivalentes Bild der selben Koordinaten ohne Waldbrand her muss. Dafür war der erste Ansatz dies mithilfe eines python scripts durchzuführen und für jede Koordinate mit Waldbrand ein Satelitenbild herunterzuladen. Dieser Ansatz ist leider gescheitert da alle Websites die getestet wurden nach ca. 10 bis 20 aufrufen der Website keine Anfragen mehr ausgegeben haben. Deswegen war die nächste Idee die Satelitenbilder mithilfe einer API von Google Earth Images herunterzuladen.


Datensammlung und Verarbeitung Sebastian:
Dieser Ansatz hat trotz einiger komplikationen sehr gut funktioniert und so existiert nun vollständiges Datenset aus den selben Koordinaten mit und ohne Waldbrand.

Da wir nun ein vollständiges Datenset haben besteht die Möglichkeit weitere GAN Ansätze zu testen.

DeepGAN Ansatz Alle:
- Erste lauffähige Implementierung
- Sicherstellung der Einhaltung der Architektur nach den im Paper ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/abs/1511.06434) genannten Vorgaben: 
  - Ersetze alle Pooling-Schichten durch Strided Convolutions (Discriminator) und Fractional-Strided Convolutions (Generator).
  - Verwende Batchnorm sowohl im Generator als auch im Diskriminator.
  - Entferne vollständig verbundene Hidden Layer für tiefere Architekturen.
  - Verwende ReLU-Aktivierung im Generator für alle Schichten außer der Ausgabe, die Tanh verwendet.
  - Verwende LeakyReLU-Aktivierung im Diskriminator für alle Schichten.


DeepGAN Fehlerbehebung Ersan:
- Analyse des Papers "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" um zu verstehen wie
die verschiedenen Techniken funktionieren und worauf man bei der Architektur achten muss bei DCGANs.
- Analyse der Funktionsweise von Strided Convolutions sowie Fractional-Strided Convolutions(Transposed Convolutions)
- Entschluss dass das Modell für die Lösung des Problems nicht geeignet ist und der Pix2Pix Ansatz eher zielführend ist

Pix2Pix Ansatz Ersan:
- Erste Implementierung
- Anpassung nach den Satelittenbilern
- Erster erfolgreicher Test
- Anpassung der Schichten damit auch wirklich die verlangte Auflösung 350x350 Pixel für jedes Bild erhalten wird
- Primitiver Ansatz bei dem ein weiteres ConvTranspose2D Layer verwendet wird fürs upscaling, gefolgt von einem resize um genau auf den Wert 350x350 zu kommen.
Der [ConvNet Calculator](https://madebyollin.github.io/convnet-calculator/) und [ConvTranspose2D Calculator](https://abdumhmd.github.io/files/convtranspose2d.html)helfen beim Verständnis des Problems von der Anpassung durch Convolutions
- Der Ansatz führte zu Problemen weil die Architektur es nicht zulässt eine weitere ConvTranspose2D hinzuzugügen.
- Anderer Ansatz mit interpolate billinear wurde verwendet -> Bilder haben die verlangte Dimension sind aber von der Qualität schlecht. Nach längerem Training muss die Qualität der Bilder getestet werden.


## Generierung von Bildern mit Pix2Pix

Laden Sie die Datei gen.pth aus unserem Teams-Kanal herutner und legen Sie sie im Ordner Test_Model ab: 
https://hawhamburgedu.sharepoint.com/:u:/s/KIPraktikumBaranFischerWewer/Ea4VlttPPwxGgvVEkoR8mSoB-1_iY-2z1aWO-nfX-1BlLg?e=tq4SF1

Um Bilder von möglichen Waldbränden zu generieren muss nur das Script [generate_image.py](Test_Model/generate_image.py) ausgeführt werden.
Das Script erstellt für jedes Satellitenbild im Ordner [Test_Model/images/original](Test_Model/images/original) ein Bild im Ordner [Test_Model/images/original](Test_Model/images/generated) auf dem die mögliche Ausbreitung eines Waldbrandes zu sehen ist.
