\# Entity Clustering and Structural Association



\## 1. Purpose



The system uses blockchain address relationships to derive structural evidence

about potentially related addresses and transactions.



The objective is not to assign real-world identities to cryptocurrency

addresses. Instead, the system identifies repeated structural relationships

that can support transaction investigation, behavioral analysis, graph

analysis, and risk scoring.



An address cluster therefore represents a structural association in the

observed transaction graph and must not be interpreted as a confirmed person,

organization, or legal entity.



\---



\## 2. Source Address Graph



The Elliptic++ actor data provides the following address-level relationships:



\- `AddrAddr\_edgelist.csv`

&#x20; - `input\_address`

&#x20; - `output\_address`



The relationship is interpreted as a directed transaction relationship from

an input address to an output address.



The raw address graph contains:



\- 2,868,964 edge observations

\- 822,942 unique addresses

\- 84,620 duplicate directed edge observations

\- 45,981 self-loop observations



The system removes duplicate directed edges while preserving self-loop

information for topology analysis.



The resulting deduplicated graph contains:



\- 2,784,344 unique directed edges

\- 822,942 unique addresses

\- 8,707 unique self-loop edges

\- 6,326 reciprocal directed edges



\---



\## 3. Global Connected Components



The deduplicated address graph was also analyzed using connected components,

with the directed graph treated as an undirected connectivity structure for

this diagnostic.



The analysis produced:



\- 2 connected components

\- 0 singleton components

\- largest component: 822,935 addresses

\- second component: 7 addresses



The dominant component is therefore too large to represent a useful final

entity identity cluster.



Consequently, global connected-component membership is retained only as

diagnostic graph information and is not used as the final entity identity

mechanism.



\---



\## 4. Transaction-Level Address Structure



The address-to-transaction relationships provide:



\- 477,117 input-address/transaction observations

\- 837,124 transaction/output-address observations

\- 400,212 unique input addresses

\- 641,043 unique output addresses

\- 202,804 transaction IDs with both input and output relationships



Observed transaction structure includes:



\- median input-address count: 1

\- maximum input-address count: 653

\- median output-address count: 2

\- maximum output-address count: 13,107

\- 43,507 transactions with more than one input address

\- 183,676 transactions with more than one output address



These relationships provide the basis for co-spend analysis.



\---



\## 5. Input Co-Sponsorship / Co-Spend Evidence



For transactions containing multiple input addresses, the system constructs

address pairs that occur together as inputs to the same transaction.



This creates an input co-spend graph.



The resulting graph contains:



\- 14,532,245 unique address pairs

\- maximum pair co-spend count: 39



Repeated co-spend evidence is evaluated at multiple thresholds:



| Threshold | Unique pairs | Non-singleton addresses |

|---|---:|---:|

| >= 1 shared transaction | 14,532,245 | 284,709 |

| >= 2 shared transactions | 242,945 | 10,518 |

| >= 3 shared transactions | 27,426 | 2,585 |



The multiple thresholds are retained as separate evidence rather than selecting

an arbitrary single threshold as a definitive identity boundary.



\---



\## 6. Entity Evidence Features



For every address, the system derives:



\- co-sponsor cluster IDs at thresholds 1, 2, and 3

\- repeated co-sponsor degree

\- maximum shared transaction count

\- repeated co-sponsor indicator

\- logarithmic repeated co-sponsor degree

\- cluster sizes at thresholds 1, 2, and 3

\- logarithmic cluster sizes at thresholds 1, 2, and 3



The resulting entity-evidence artifact contains:



\- 822,942 addresses

\- 14 columns

\- 10,518 addresses with repeated co-sponsor evidence

\- maximum repeated co-sponsor degree: 525

\- maximum shared transaction count: 39



\---



\## 7. Transaction-Level Entity Evidence



Address-level evidence is aggregated back to individual transactions.



For both input and output sides, the system records:



\- address count

\- repeated co-sponsor count

\- repeated co-sponsor ratio

\- maximum repeated co-sponsor degree

\- maximum shared transaction count

\- distinct candidate clusters at thresholds 1, 2, and 3

\- maximum candidate cluster size at thresholds 1, 2, and 3



The resulting artifact contains:



\- 203,769 transactions

\- 23 columns including `txid`

\- 203,769 unique transaction IDs

\- 0 null values

\- 0 infinite values



Measured transaction coverage:



\- 13,378 transactions contain input-side repeated co-sponsor evidence

\- 23,794 transactions contain output-side repeated co-sponsor evidence



\---



\## 8. Integration With the ML Feature Matrix



The transaction-level entity evidence is integrated with:



1\. transaction features

2\. temporal features

3\. wallet structural features

4\. point-in-time wallet history features

5\. graph/entity evidence



The resulting unified feature matrix contains:



\- 203,769 transactions

\- 93 columns

\- 92 analytical features plus `txid`

\- 0 duplicate TXIDs

\- 0 duplicate column names

\- 0 infinite values



The existing 965 transactions with incomplete source transaction/address

relationships retain their structured missing values. No blanket imputation is

performed during feature construction.



\---



\## 9. Interpretation and Limitations



The entity-clustering subsystem identifies structural relationships, not

real-world identities.



A repeated co-spend relationship can provide evidence that addresses have

appeared together in transaction inputs, but it does not independently prove

common ownership or control.



Similarly:



\- a graph cluster is not a person;

\- an address is not automatically an identified entity;

\- structural association is not attribution;

\- transaction risk is not proof of criminal activity.



Any real-world attribution must require independent investigative evidence

outside the blockchain graph.



\---



\## 10. Role in the Final System



Entity evidence is used as supporting intelligence for:



\- entity-level behavioral analysis

\- transaction anomaly detection

\- graph investigation

\- risk scoring

\- alert explanation

\- analyst-facing link analysis



It is not used as a hard identity label.



The final system combines this structural evidence with transaction behavior,

temporal behavior, network metadata, anomaly detection, supervised ML, and

explainability to produce ranked investigative alerts.
