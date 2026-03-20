class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.6.1/machinator-0.6.1.tar.gz"
  sha256 "c994b68d28396329688fdcd39457bc2515dd3e73173153821efed4f0a36d3d13"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/macht --help")
  end
end
