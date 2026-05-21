For Developers
============
You can also see [C++](https://github.com/StarlangSoftware/SequenceProcessing-CPP), or [C#](https://github.com/StarlangSoftware/SequenceProcessing-CS) repository.
## Requirements

* [Java Development Kit 8 or higher](#java), Open JDK or Oracle JDK
* [Maven](#maven)
* [Git](#git)

### Java 

To check if you have a compatible version of Java installed, use the following command:

    java -version
    
If you don't have a compatible version, you can download either [Oracle JDK](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) or [OpenJDK](https://openjdk.java.net/install/)    

### Maven
To check if you have Maven installed, use the following command:

    mvn --version
    
To install Maven, you can follow the instructions [here](https://maven.apache.org/install.html).      

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called StructureConverter will be created. Or you can use below link for exploring the code:

	git clone https://github.com/starlangsoftware/SequenceProcessing.git

## Open project with IntelliJ IDEA

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `SequenceProcessing/pom.xml` file
* Select open as project option
* Couple of seconds, dependencies with Maven will be downloaded. 


## Compile

**From IDE**

After being done with the downloading and Maven indexing, select **Build Project** option from **Build** menu. After compilation process, user can run SequenceProcessing.

**From Console**

Go to `SequenceProcessing` directory and compile with 

     mvn compile 

## Generating jar files

**From IDE**

Use `package` of 'Lifecycle' from maven window on the right and from `SequenceProcessing` root module.

**From Console**

Use below line to generate jar file:

     mvn install

## Maven Usage

        <dependency>
            <groupId>io.github.starlangsoftware</groupId>
            <artifactId>SequenceProcessing</artifactId>
            <version>1.0.0</version>
        </dependency>

For Contibutors
============

### pom.xml file
1. Standard setup for packaging is similar to:
```
    <groupId>io.github.starlangsoftware</groupId>
    <artifactId>Amr</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    <name>NlpToolkit.Amr</name>
    <description>Abstract Meaning Representation Library</description>
    <url>https://github.com/StarlangSoftware/Amr</url>

    <organization>
        <name>io.github.starlangsoftware</name>
        <url>https://github.com/starlangsoftware</url>
    </organization>

    <licenses>
        <license>
            <name>The Apache Software License, Version 2.0</name>
            <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
        </license>
    </licenses>

    <developers>
        <developer>
            <name>Olcay Taner Yildiz</name>
            <email>olcay.yildiz@ozyegin.edu.tr</email>
            <organization>Starlang Software</organization>
            <organizationUrl>http://www.starlangyazilim.com</organizationUrl>
        </developer>
    </developers>

    <scm>
        <connection>scm:git:git://github.com/starlangsoftware/amr.git</connection>
        <developerConnection>scm:git:ssh://github.com:starlangsoftware/amr.git</developerConnection>
        <url>http://github.com/starlangsoftware/amr/tree/master</url>
    </scm>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
```
2. Only top level dependencies should be added. Do not forget junit dependency.
```
    <dependencies>
        <dependency>
            <groupId>io.github.starlangsoftware</groupId>
            <artifactId>AnnotatedSentence</artifactId>
            <version>1.0.78</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.1</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
```
3. Maven compiler, gpg, source, javadoc plugings should be added.
```
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-compiler-plugin</artifactId>
		<version>3.6.1</version>
		<configuration>
			<source>1.8</source>
			<target>1.8</target>
		</configuration>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-gpg-plugin</artifactId>
		<version>1.6</version>
		<executions>
			<execution>
				<id>sign-artifacts</id>
				<phase>verify</phase>
				<goals>
					<goal>sign</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-source-plugin</artifactId>
		<version>2.2.1</version>
		<executions>
			<execution>
				<id>attach-sources</id>
				<goals>
					<goal>jar-no-fork</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-javadoc-plugin</artifactId>
		<configuration>
			<source>8</source>
		</configuration>
		<version>3.10.0</version>
		<executions>
			<execution>
				<id>attach-javadocs</id>
				<goals>
					<goal>jar</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
```
4. Currently publishing plugin is Sonatype.
```
	<plugin>
		<groupId>org.sonatype.central</groupId>
		<artifactId>central-publishing-maven-plugin</artifactId>
		<version>0.8.0</version>
		<extensions>true</extensions>
		<configuration>
			<publishingServerId>central</publishingServerId>
			<autoPublish>true</autoPublish>
		</configuration>
	</plugin>
```
5. For UI jar files use assembly plugins.
```
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-assembly-plugin</artifactId>
		<version>2.2-beta-5</version>
		<executions>
			<execution>
				<id>sentence-dependency</id>
				<phase>package</phase>
				<goals>
					<goal>single</goal>
				</goals>
				<configuration>
					<archive>
						<manifest>
							<mainClass>Amr.Annotation.TestAmrFrame</mainClass>
						</manifest>
					</archive>
					<finalName>amr</finalName>
				</configuration>
			</execution>
		</executions>
		<configuration>
			<descriptorRefs>
				<descriptorRef>jar-with-dependencies</descriptorRef>
			</descriptorRefs>
			<appendAssemblyId>false</appendAssemblyId>
		</configuration>
	</plugin>
```
### Resources
1. Add resources to the resources subdirectory. These will include image files (necessary for UI), data files, etc.
   
### Java files
1. Do not forget to comment each function.
```
    /**
     * Returns the value of a given layer.
     * @param viewLayerType Layer for which the value questioned.
     * @return The value of the given layer.
     */
    public String getLayerInfo(ViewLayerType viewLayerType){
```
2. Function names should follow caml case.
```
    public MorphologicalParse getParse()
```
3. Write toString methods, if necessary.
4. Use Junit for writing test classes. Use test setup if necessary.
```
public class AnnotatedSentenceTest {
    AnnotatedSentence sentence0, sentence1, sentence2, sentence3, sentence4;
    AnnotatedSentence sentence5, sentence6, sentence7, sentence8, sentence9;

    @Before
    public void setUp() throws Exception {
        sentence0 = new AnnotatedSentence(new File("sentences/0000.dev"));
```

# BERT Implementation (Vagif Ismayilov, S027328)

This fork contains my implementation of BERT (Bidirectional Encoder Representations from Transformers) in Java as part of the NLP course project at Ozyegin University.

## Project Status: COMPLETED
The architecture is fully implemented, verified, and integrated with the Turkish morphological analyzer library. The pipeline successfully executes automated vocabulary building, text-to-tensor conversion, and multi-epoch model training with weight serialization.

## 🛠 Features & Architecture Implemented

### 1. Hybrid Turkish Tokenizer (`HybridTokenizer`)
* Built strictly based on the paper *"Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"*.
* Integrated with the **Starlang Morphological Analyzer** (`FsmMorphologicalAnalyzer`) to accurately resolve Turkish grammar by splitting surface forms into exact roots and grammatical sub-word tags (e.g., `##prog`, `##1sg`).
* Supported by an automated **`VocabBuilder`** script that parses raw corpora (`atis-tr.txt`, `atis-en.txt`) to dynamically generate a highly optimized `vocab_real.txt` containing **1,654 unique linguistic tokens**.

### 2. Computational Graph BERT Model (`BertModel`)
A complete Transformer-based encoder architecture built utilizing the university's core mathematical framework:
* **Multi-Head Attention Node:** Parallel attention heads calculating sequence dependencies.
* **Feed-Forward Network (FFN):** Position-wise dense layers with `Tanh` activation functions.
* **Layer Normalization & Residual Connections:** Explicitly structured to ensure stable gradient flow during backpropagation.

### 3. Automated Training Pipeline & Serialization
* Automated script reads multi-language corpora, converts string data into numerical Token IDs, and structures them into multidimensional **`Math.Tensor`** objects aligned to `[SequenceLength, HiddenSize]`.
* Complete training loop executing **10 full epochs** over the dataset, calculating forward and backward derivatives via `backpropagation()`.
* **Model Serialization:** Automatically dumps and compresses the fully trained model weights into a binary deployment file (`bert_weights.model`) upon successful execution.

## 📦 Verified Environment & Dependencies
* **Java Version:** JDK 25
* **IDE:** IntelliJ IDEA 2025.2.3
* **Core Libraries Managed:**
    * `MorphologicalAnalysis` (v1.0.62)
    * `Dictionary` (v1.0.35) — *Manually upgraded to resolve transitively inherited dependency conflicts (`NoSuchMethodError`).*
    * `ComputationalGraph` (v1.0.19)
    * `Math` (v1.0.15)

## 💻 How to Run & Verify

1. Ensure all standard Turkish corpus files (`atis-tr.txt`, etc.) are placed in the root directory.
2. Open the project in IntelliJ IDEA and trigger a **Maven Reload** to fetch the updated `Dictionary` library.
3. Open `SequenceProcessing.Classification.BertMain` and click **Run**.