# Theoretical Foundations

## Idea

OpenNeuro models real-time data processing as a network of composable components
with typed interfaces. This page develops the categorical semantics of this system
incrementally.

## 1. Categories

### Definition 1.1 (Category)

A **category** **C** consists of:

1. A collection of **objects** $\text{Ob}(\mathbf{C})$.
2. For every pair of objects $A, B$, a collection of **morphisms** $\text{Hom}(A, B)$.
3. For every object $A$, an **identity morphism** $\text{id}_A \in \text{Hom}(A, A)$.
4. For every triple $A, B, C$ and morphisms $f \in \text{Hom}(A, B)$, $g \in \text{Hom}(B, C)$,
   a **composition** $g \circ f \in \text{Hom}(A, C)$.

subject to:

- **Associativity.** For $f : A \to B$, $g : B \to C$, $h : C \to D$:

$$h \circ (g \circ f) = (h \circ g) \circ f$$

- **Identity.** For $f : A \to B$:

$$f \circ \text{id}_A = f = \text{id}_B \circ f$$

Roughly, in Python:

```python
class Category[A, B](Protocol):
    @staticmethod
    def id[X]() -> Category[X, X]: ...
    def compose[C](self, other: Category[B, C]) -> Category[A, C]: ...
    # self.compose(other.compose(h)) == self.compose(other).compose(h)
    # id().compose(f) == f == f.compose(id())
```

### Remark 1.2

A category is the minimal structure needed to talk about "things that compose."
This is required to describe the syntax of OpenNeuro's graphical editor, since
components can compose to form graphs.
### Example 1.3 (The category of frame types and components)

We construct a simple category **Proc** whose:

- **Objects** are frame types: $\text{Audio}$, $\text{Text}$, $\text{Video}$,
  $\text{Interrupt}$, $\text{BodyPose}$.
- **Morphisms** are components. Each component $f$ has a typed interface $f : A \to B$,
  meaning it receives frames of type $A$ and produces frames of type $B$. The morphisms
  of **Proc** are the following components:

$$\text{ASR} : \text{Audio} \to \text{Text}$$
$$\text{TTS} : \text{Text} \to \text{Audio}$$
$$\text{LLM} : \text{Text} \to \text{Text}$$

the identity (passthrough) morphisms for each object, e.g.
$\text{id}_{\text{Audio}} : \text{Audio} \to \text{Audio}$, and anything that can be
formed by composing these.

Composition is wiring: given $\text{ASR} : \text{Audio} \to \text{Text}$ and
$\text{LLM} : \text{Text} \to \text{Text}$, their composition is:

$$\text{LLM} \circ \text{ASR} : \text{Audio} \to \text{Text}$$

### Definition 1.4 (Free category)

Given a directed graph $G$ with nodes (objects) and edges (primitive morphisms), the
**free category** $F(G)$ is the category whose morphisms are all finite directed paths
in $G$, composed by path concatenation, with the empty path as identity.

No equations are imposed beyond the category axioms. Two morphisms are equal if and
only if they are the same sequence of primitive morphisms. This means every distinct
way of composing primitives produces a distinct morphism — the free category is the
space of all programs that can be constructed from the given primitives.

See [nLab: free category](https://ncatlab.org/nlab/show/free+category).

### Remark 1.5

**Proc** as constructed in Example 1.3 is a free category, where the primitive
morphisms are the components and the objects are the frame types.

### Remark 1.6 (Limitation)

Consider a graph where $\text{ASR} : \text{Audio} \to \text{Text}$ and
$\text{PoseRenderer} : \text{BodyPose} \to \text{Video}$ run side by side with no
connection between them. This cannot be expressed in **Proc** — composition ($\circ$)
requires the output of one morphism to match the input of the next. There is no
operation for placing two independent morphisms in parallel. To model this, we need
additional structure.

## 2. Commutative monoidal categories

### Definition 2.1 (Monoidal category)

A **monoidal category** $(\mathbf{C}, \otimes, I)$ is a category $\mathbf{C}$ equipped with:

1. A bifunctor $\otimes : \mathbf{C} \times \mathbf{C} \to \mathbf{C}$, called the
   **tensor product** or **monoidal product**.
2. A **unit object** $I$.
3. Natural isomorphisms:
   - **Associator:** $\alpha_{A,B,C} : (A \otimes B) \otimes C \xrightarrow{\sim} A \otimes (B \otimes C)$
   - **Left unitor:** $\lambda_A : I \otimes A \xrightarrow{\sim} A$
   - **Right unitor:** $\rho_A : A \otimes I \xrightarrow{\sim} A$

satisfying the triangle and pentagon coherence conditions (see
[nLab: monoidal category](https://ncatlab.org/nlab/show/monoidal+category)).

### Definition 2.2 (Symmetric monoidal category)

A **symmetric monoidal category** is a monoidal category equipped with a natural
isomorphism (the **braiding**):

$$\sigma_{A,B} : A \otimes B \xrightarrow{\sim} B \otimes A$$

such that $\sigma_{B,A} \circ \sigma_{A,B} = \text{id}_{A \otimes B}$.

### Definition 2.3 (Commutative monoidal category)

A monoidal category is **commutative** (also called **strictly symmetric**) when the
braiding is the identity:

$$A \otimes B = B \otimes A$$

That is, the tensor product is commutative on the nose, not merely up to isomorphism.

### Remark 2.4

A commutative monoidal category is a symmetric monoidal category where the symmetry
is trivial. This is a stronger condition than mere symmetry.

### Remark 2.5 (Interfaces)

In the simple **Proc** of §1, objects are bare frame types like $\text{Audio}$ or
$\text{Text}$. In practice, components often have multiple input or output ports. For
example, an LLM component takes both text and an interrupt signal:

$$\text{LLM} : (\text{text}: \text{Text},\; \text{interrupt}: \text{Interrupt}) \to \text{Text}$$

We therefore refine our notion of object: objects in **Proc** are **interfaces** —
named records of typed ports. A single-port interface like $\text{Audio}$ is just a
record with one field. Multi-port interfaces arise naturally from the components
themselves, not only from parallel composition.

### Remark 2.6 (Parallel composition in Proc)

In OpenNeuro, the tensor product is commutative on the nose at both levels:

- **Objects.** Component interfaces use **named ports** (Python `NamedTuple` fields),
  not positionally ordered tuples. The interface `(audio: Audio, video: Video)` is the
  same object as `(video: Video, audio: Audio)` — the names determine identity, not
  the order. So $A \otimes B = B \otimes A$.

- **Morphisms.** Parallel execution has no notion of "first" or "second" — both
  components run independently on their own threads. So $f \otimes g = g \otimes f$.

This makes **Proc** a commutative monoidal category with:

- $A \otimes B$ = the combined interface with all ports from $A$ and $B$.
- $I$ = the empty interface (a component with no inputs or no outputs).

### Example 2.7 (Parallel composition in Proc)

Recall from Remark 1.6 that $\text{ASR}$ and $\text{PoseRenderer}$ running side by side
could not be expressed in a plain category. With the monoidal product, we can now write:

$$\text{ASR} \otimes \text{PoseRenderer} : \text{Audio} \otimes \text{BodyPose} \to \text{Text} \otimes \text{Video}$$

Both components run in parallel. The composite interface has two input ports
($\text{Audio}$, $\text{BodyPose}$) and two output ports ($\text{Text}$, $\text{Video}$).

### Remark 2.8 (Relation to Haskell's Arrow)

The `Arrow` type class in Haskell corresponds closely to a (strict) monoidal category.
The operations are:

| Haskell Arrow | Categorical |
|---|---|
| `arr f` | lifting a pure function to a morphism |
| `>>>` | sequential composition ($\circ$) |
| `***` | parallel composition ($\otimes$) |
| `first f` | $f \otimes \text{id}$ |

See [nLab: Arrow](https://ncatlab.org/nlab/show/Arrow).

### Remark 2.9 (Limitation)

Consider an $\text{AgentState} : \text{Text} \to \text{History}$ component that
accumulates conversation history, and an $\text{LLM} : \text{History} \to \text{Text}$
component. Composing them sequentially gives:

$$\text{LLM} \circ \text{AgentState} : \text{Text} \to \text{Text}$$

But we want the $\text{Text}$ output of $\text{LLM}$ to feed back into
$\text{AgentState}$, forming a loop — the agent responds, the response is appended to
history, and the updated history feeds the next LLM call. A commutative monoidal
category has no operation for this — there is no way to wire an output back into an
input. To model feedback, we need trace.

## 3. Traced commutative monoidal categories

### Definition 3.1 (Trace)

Let $(\mathbf{C}, \otimes, I)$ be a symmetric monoidal category. A **trace** on
$\mathbf{C}$ is a family of functions:

$$\text{Tr}^X_{A,B} : \text{Hom}(A \otimes X, \; B \otimes X) \to \text{Hom}(A, B)$$

for every triple of objects $A, B, X$, satisfying naturality and coherence axioms
(see [nLab: traced monoidal category](https://ncatlab.org/nlab/show/traced+monoidal+category)).

### Remark 3.2

Intuitively, $\text{Tr}^X_{A,B}$ takes a morphism that has an "extra" input and output
of type $X$ and produces a morphism where $X$ is wired internally as a feedback loop.
The external interface shrinks from $(A \otimes X) \to (B \otimes X)$ to just $A \to B$.

### Example 3.3 (Feedback in Proc)

Recall from Remark 2.9 the $\text{AgentState} : \text{Text} \to \text{History}$ and
$\text{LLM} : \text{History} \to \text{Text}$ components. Their composition
$\text{LLM} \circ \text{AgentState} : \text{Text} \to \text{Text}$ is a morphism
with matching input and output types. We can model this as a morphism with an extra
feedback channel:

$$P : \text{Text} \otimes \text{Text} \to \text{Text} \otimes \text{Text}$$

Applying trace over $\text{Text}$:

$$\text{Tr}^{\text{Text}}(P) : \text{Text} \to \text{Text}$$

The feedback channel becomes internal — the LLM's text output feeds back into
AgentState, whose history output feeds the next LLM call. From the outside, it is
simply a $\text{Text} \to \text{Text}$ component.

### Remark 3.4 (Relation to Haskell's ArrowLoop)

The `ArrowLoop` type class adds feedback to `Arrow`:

```haskell
class Arrow a => ArrowLoop a where
    loop :: a (b, d) (c, d) -> a b c
```

This is exactly the trace operator: `loop` takes a morphism of type $(B \otimes D) \to (C \otimes D)$
and produces one of type $B \to C$ by feeding $D$ back. The `rec` keyword in arrow notation
is syntactic sugar for `loop`.

### Remark 3.5 (Limitation)

Consider an $\text{ASR} : \text{Audio} \to \text{Text}$ whose text output should feed
both an $\text{LLM}$ and a $\text{Logger}$ simultaneously. Or two independent
$\text{ASR}$ components whose text outputs should merge into a single $\text{LLM}$
input. A traced commutative monoidal category has no operation for either — every
output goes to exactly **one** input. We need:

- **Fan-out (copying):** one output feeds multiple downstream components.
- **Fan-in (merging):** multiple outputs feed into one input.
- **Discard:** an output that goes nowhere.

## 4. Copy, merge, and discard

### Definition 4.1 (Commutative comonoid)

Let $(\mathbf{C}, \otimes, I)$ be a monoidal category. A **commutative comonoid** on
an object $A$ is a pair of morphisms:

$$\Delta_A : A \to A \otimes A \quad \text{(copy)}$$
$$\varepsilon_A : A \to I \quad \text{(discard)}$$

satisfying coassociativity, counitality, and commutativity (see
[nLab: comonoid](https://ncatlab.org/nlab/show/comonoid)).

### Definition 4.2 (Commutative monoid)

Dually, a **commutative monoid** on an object $A$ is a pair of morphisms:

$$\nabla_A : A \otimes A \to A \quad \text{(merge)}$$
$$\eta_A : I \to A \quad \text{(create)}$$

satisfying associativity, unitality, and commutativity (see
[nLab: monoid](https://ncatlab.org/nlab/show/monoid+in+a+monoidal+category)).

### Definition 4.3 (gs-monoidal category)

A **gs-monoidal category** (garbage-sharing monoidal category) is a symmetric monoidal
category in which every object is equipped with a commutative comonoid structure
$(\Delta_A, \varepsilon_A)$ that is natural in $A$.

"Garbage" refers to discard ($\varepsilon$), and "sharing" refers to copy ($\Delta$).

See [Corradini & Gadducci, 1999] for the original definition.

### Definition 4.4 (gs-monoidal category with merges)

We extend the gs-monoidal structure by additionally equipping every object with a natural
commutative monoid structure $(\nabla_A, \eta_A)$. This gives every object the structure
of a **commutative bimonoid**: both a commutative monoid and a commutative comonoid.

### Example 4.5 (Fan-out, fan-in, and discard in Proc)

In OpenNeuro, these operations are provided by the channel system:

- **Copy ($\Delta$).** A `Sender` holds a list of `Channel`s. When it sends a frame,
  every channel receives a copy. This is fan-out: one output going to multiple
  downstream components.

- **Discard ($\varepsilon$).** A `Sender` with no connected channels. The frame is
  produced but nobody reads it. The data is silently discarded.

- **Merge ($\nabla$).** Multiple `Sender`s connected to the same `Channel`. Frames
  from different upstream components are interleaved by arrival order into a single
  stream.

- **Create ($\eta$).** A source component (e.g. `Microphone`) that produces frames
  from nothing — it has no input channels, only outputs.

These operations are inherent to the channel design, not bolted on. Any component's
output can fan out to arbitrarily many downstream components, and any input can receive
from arbitrarily many upstream components, without any special wiring logic.

## 5. The type lattice

### Definition 5.1 (Partial order on objects)

The objects of **Proc** carry a **partial order** $\leq$ given by the subtyping
relation on frame types. We write $A \leq B$ when $A$ is a subtype of $B$.

In OpenNeuro, this is the class hierarchy:

```
            Frame
          / |  |  \
   Audio  Text  Video  BodyPose  ...
           |
          EOS
```

So $\text{EOS} \leq \text{Text} \leq \text{Frame}$, but $\text{Audio}$ and
$\text{Video}$ are incomparable.

### Remark 5.2 (Wiring compatibility)

A connection from a $\text{Sender}[A]$ to a $\text{Receiver}[B]$ is valid when
$A \leq B$. This means a component that outputs $\text{EOS}$ frames can wire into
any input that accepts $\text{Text}$, because $\text{EOS} \leq \text{Text}$.

The partial order gives rise to implicit coercion morphisms $\iota_{A,B} : A \to B$
for every $A \leq B$. These are the "silent" type conversions that the system handles
without the user writing an explicit adapter component.

### Remark 5.3

In categorical terms, the partial order makes the collection of objects a
[poset](https://ncatlab.org/nlab/show/partial+order), and the coercion morphisms
are the unique morphisms guaranteed by the partial order. This enriches our category
with a subtyping discipline: composition respects the order, and the monoidal product
extends it componentwise.

## 6. Higher-order components

### Definition 6.1 (Functor)

A **functor** $F : \mathbf{C} \to \mathbf{D}$ between two categories consists of:

1. A mapping on objects: $A \mapsto F(A)$.
2. A mapping on morphisms: $(f : A \to B) \mapsto (F(f) : F(A) \to F(B))$.

preserving identity and composition:

$$F(\text{id}_A) = \text{id}_{F(A)}$$
$$F(g \circ f) = F(g) \circ F(f)$$

An **endofunctor** is a functor from a category to itself: $F : \mathbf{C} \to \mathbf{C}$.

See [nLab: functor](https://ncatlab.org/nlab/show/functor).

### Example 6.2 (Generic components as endofunctors)

A component that is **generic over its frame type** defines an endofunctor on **Proc**.
For instance, a `Passthrough[T]` component — parameterized by a type $T$ — maps:

- Each object $T$ to itself.
- Each morphism $f : A \to B$ to a morphism $\text{Passthrough}(f) : A \to B$ that applies
  $f$ while passing data through.

More practical examples of endofunctors on **Proc** include:

- A **profiler** that wraps any component and adds timing measurements, without changing
  its interface type.
- A **retry wrapper** that takes any component $f : A \to B$ and produces a fault-tolerant
  version $\text{Retry}(f) : A \to B$.

These are structure-preserving: wrapping a composition is the same as composing the
wrapped versions. That is precisely the functor law $F(g \circ f) = F(g) \circ F(f)$.

## 7. Putting it together

### Remark 7.1

Extending the free category of Definition 1.4 with all the structure introduced in
§2–§4, OpenNeuro's graph language generates a **free traced commutative gs-monoidal
category with merges** — the free category built from the declared components as
primitive morphisms, closed under sequential composition ($\circ$), parallel composition
($\otimes$), trace ($\text{Tr}$), copy ($\Delta$), merge ($\nabla$), and discard
($\varepsilon$), with no additional equations.

Every morphism in this free category corresponds to a graph that can be constructed in
the system. Every constructible graph corresponds to a morphism. The free category is
exactly the space of all programs expressible in the system.

### Definition 7.2 (Interpretation)

An **interpretation** of the free category is a structure-preserving functor:

$$\llbracket - \rrbracket : F(\text{Generators}) \to \mathbf{Sem}$$

from the free category into a **semantic category** $\mathbf{Sem}$ where morphisms have
computational meaning (e.g., actual running threads passing real frames through channels).

The interpretation functor must preserve all structure:

- $\llbracket g \circ f \rrbracket = \llbracket g \rrbracket \circ \llbracket f \rrbracket$
- $\llbracket f \otimes g \rrbracket = \llbracket f \rrbracket \otimes \llbracket g \rrbracket$
- $\llbracket \text{Tr}^X(f) \rrbracket = \text{Tr}^{\llbracket X \rrbracket}(\llbracket f \rrbracket)$

Because the source is free, the interpretation is **uniquely determined** by its action
on the generators. Once you assign a concrete implementation to each primitive component,
the semantics of every composite graph is fixed automatically by functoriality.

### Remark 7.3 (Abstract components)

A generator in the free category that has not been assigned an interpretation is an
**abstract component** — declared with a typed interface but no implementation. A graph
containing abstract components cannot be fully interpreted (executed) until every
abstract generator is given a concrete meaning.

This is the categorical account of "plug-in" or "placeholder" components: the graph
structure is well-defined regardless of whether implementations exist, but execution
requires a complete interpretation.

## Summary

The following table summarizes the correspondence between OpenNeuro concepts and their
categorical counterparts.

| OpenNeuro | Category theory | Introduced in |
|---|---|---|
| Frame types (`Audio`, `Text`, `Video`, ...) | Objects | §1 |
| Components (`Component[I, O]`) | Morphisms | §1 |
| Wiring output to input | Sequential composition ($\circ$) | §1 |
| All constructible graphs | Free category | §1 |
| Running side by side | Parallel composition ($\otimes$) | §2 |
| Named ports (order doesn't matter) | Commutativity of $\otimes$ | §2 |
| Output wired back to input | Trace ($\text{Tr}^X$) | §3 |
| Fan-out (one output, many inputs) | Copy ($\Delta$) | §4 |
| Unused output | Discard ($\varepsilon$) | §4 |
| Fan-in (many outputs, one input) | Merge ($\nabla$) | §4 |
| Source component (no inputs) | Create ($\eta$) | §4 |
| Frame subtyping (`EOS ≤ Text`) | Partial order on objects | §5 |
| Generic / higher-order component | Endofunctor | §6 |
| Runtime execution | Interpretation functor ($\llbracket - \rrbracket$) | §7 |

## References

- [nLab: category](https://ncatlab.org/nlab/show/category)
- [nLab: monoidal category](https://ncatlab.org/nlab/show/monoidal+category)
- [nLab: symmetric monoidal category](https://ncatlab.org/nlab/show/symmetric+monoidal+category)
- [nLab: traced monoidal category](https://ncatlab.org/nlab/show/traced+monoidal+category)
- [nLab: comonoid](https://ncatlab.org/nlab/show/comonoid)
- [nLab: monoid in a monoidal category](https://ncatlab.org/nlab/show/monoid+in+a+monoidal+category)
- [nLab: free category](https://ncatlab.org/nlab/show/free+category)
- [nLab: functor](https://ncatlab.org/nlab/show/functor)
- Corradini, A. & Gadducci, F. (1999). "An algebraic presentation of term graphs, via gs-monoidal categories."
- Hasegawa, M. (1997). "Recursion from cyclic sharing: traced monoidal categories and models of cyclic lambda calculi."
